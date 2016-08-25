import numpy 
import skylark
from ctypes import byref, c_void_p, c_double, c_int
from skylark import sketch, errors
from distances import euclidean
import scipy.special
import scipy
import sys, math
import skylark.lib as lib

# Kernel factory
def kernel(kerneltype, **kwargs):
  """
  Returns a kernel based on the input parameters.

  :param kerneltype: string identifying the kernel requested.
  :param d: dimension of the kernel.
  :param **kwargs: dictonary of kernel parameters, kernel dependent.
  :returns: kernel object
  """
  if not isinstance(kerneltype, str):
    raise ValueError("kerneltype must be a string")
  elif kerneltype.lower() == "linear":
    return Linear(d, **kwargs)
  elif kerneltype.lower() == "matern":
    return Matern(**kwargs)
  elif kerneltype.lower() == "gaussian":
    return Gaussian(**kwargs)
  else:
    raise ValueError("kerneltype not recognized")


# Kernel base
class Kernel(object):
  """
  TODO: Base kernel
  """

  def __get_direction(self, dir):
    """
    Returns the direction of dir (1 or 2)

    :param dir: 0/1 or "rows"/"cols"
    """
    if dir == 0 or dir == "columns":
      return 1
    elif dir == 1 or dir == "rows":
      return 2
    else:
      raise ValueError("Direction must be either columns/rows or 0/1")


  def gram(self, X, K, dirX="rows", dirY="rows", Y=None):
    """
    Returns the dense Gram matrix evaluated over the datapoints.
  
    :param X: n-by-d data matrix
    :param Y: another data matrix. If Y is None, then X is used.
    :param K: placeholder for output Gram matrix.
    """

    if Y is None:
      Y = X
      dirY = dirX

    cdirX = self.__get_direction(dirX)
    cdirY = self.__get_direction(dirY)

    X = lib.adapt(X)
    Y = lib.adapt(Y)
    K = lib.adapt(K)

    lib.callsl("sl_kernel_gram", cdirX, cdirY, self._kernel_obj, \
                X.ctype(), X.ptr(), \
                Y.ctype(), Y.ptr(), \
                K.ctype(), K.ptr())

    X.ptrcleaner()
    Y.ptrcleaner()
    K.ptrcleaner()

    return K.getobj()




# Kernel implementations

class Linear(Kernel):
  """
  A object representing the Linear kernel over d dimensional vectors.

  :param d: dimension of vectors on which kernel operates.
  """

  def __init__(self, d):
    self._d = d
    
  def gram(self, X, Xt=None):
    """
    Returns the dense Gram matrix evaluated over the datapoints.
  
    :param X:  n-by-d data matrix
    :param Xt: optional t-by-d test matrix

    Returns: 
    -------
    n-by-n Gram matrix over X (if Xt is not provided)
    t-by-n Gram matrix between Xt and X if X is provided
    """
  
    # TODO the test, and this function, should work for all matrix types.
    if X.shape[1] != self._d:
      raise ValueError("X must have vectors of dimension d")

    if Xt is None:
      K = numpy.dot(X, X.T)
    else:
      if Xt.shape[1] != self._d:
        raise ValueError("Xt must have vectors of dimension d")
      K = numpydot(Xt, X.T)
      
    return K
  
  def rft(self, s, subtype=None, defouttype=None, **kwargs):
    """
    Create a random features transform for the kernel.
    This function uses random Fourier features (Rahimi-Recht).
    
    :param s: number of random features.
    :param subtype: subtype of rft to use. Accepted values:
           None - will default to JLT.
           fast - will use FJLT
           hash - will use CWT
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """
    if subtype is None:
      return sketch.JLT(self._d, s, defouttype, **kwargs)
    elif subtype is 'fast':
      return sketch.FJLT(self._d, s, defouttype, **kwargs)
    elif subtype is 'hash':
      return skethc.CWT(self._d, s, defouttype, **kwargs)
    else:
      raise ValueError("invalide subtype supplied")


class Matern(object):
  """
  A object representing the Matern kernel over d dimensional vectors, with
  nu and l

  :param d: dimension of vectors on which kernel operates.
  :param nu: nu parameter
  :param l: l parameter
  """

  def __init__(self, d, nu, l):
    self._d = d
    self._nu = nu
    self._l = l
    
  def gram(self, X, Xt=None):
    """
    Returns the dense Gram matrix evaluated over the datapoints.
  
    :param X:  n-by-d data matrix
    :param Xt: optional t-by-d test matrix

    Returns: 
    -------
    n-by-n Gram matrix over X (if Xt is not provided)
    t-by-n Gram matrix between Xt and X if X is provided
    """
  
    # TODO the test, and this function, should work for all matrix types.
    if X.shape[1] != self._d:
      raise ValueError("X must have vectors of dimension d")

    nu = self._nu
    l = self._l
    if Xt is None:
        D = euclidean(X, X)
    else:
        if Xt.shape[1] != self._d:
            raise ValueError("Xt must have vectors of dimension d")
        D = euclidean(X, Xt);

    Y = scipy.sqrt(2.0 * nu * D) / l
    K = 2.0 ** (1 - nu) / scipy.special.gamma(nu) * Y ** nu * scipy.special.kv(nu, Y)

    return scipy.real(K)
  
  def rft(self, s, subtype=None, defouttype=None, **kargs):
    """
    Create a random features transform for the kernel.
    This function uses random Fourier features (Rahimi-Recht).
    
    :param s: number of random features.
    :param subtype: subtype of rft to use (e.g. sparse, fast).
           Currently we support regular (None) and fast.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """
    return sketch.MaternRFT(self._d, s, self._nu, self._l, defouttype, **kargs)


# Kernel bindings

class Gaussian(Kernel):
  """
  A object representing the Gaussian kernel over d dimensional vectors, with
  bandwidth sigma.

  :param d: dimension of vectors on which kernel operates.
  :param sigma: bandwidth of the kernel.
  """

  def __init__(self, d, sigma):
    self._d = d
    self._sigma = sigma
    self._kernel_obj = c_void_p()
    lib.callsl("sl_create_kernel", "gaussian", d, byref(self._kernel_obj), c_double(sigma))

  def rft(self, s, subtype=None, defouttype=None, **args):
    """
    Create a random features transform for the kernel.
    This function uses random Fourier features (Rahimi-Recht).
    
    :param s: number of random features.
    :param subtype: subtype of rft to use (e.g. sparse, fast).
           Currently we support regular (None) and fast.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """
    if subtype is 'fast':
      return sketch.FastGaussianRFT(self._d, s, self._sigma, defouttype, **args)
    else:
      return sketch.GaussianRFT(self._d, s, self._sigma, defouttype, **args)
    