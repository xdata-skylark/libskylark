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
def kernel(kerneltype, *args):
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
    return Linear(d, *args)
  elif kerneltype.lower() == "matern":
    return Matern(*args)
  elif kerneltype.lower() == "expsemigroup":
    return ExpSemiGroup(*args)
  elif kerneltype.lower() == "gaussian":
    return Gaussian(*args)
  elif kerneltype.lower() == "polynomial":
    return Polynomial(*args)
  elif kerneltype.lower() == "laplacian":
    return Laplacian(*args)
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



# Kernel bindings

class Linear(Kernel):
  """
  A object representing the Linear kernel over d dimensional vectors.

  :param d: dimension of vectors on which kernel operates.
  """

  def __init__(self, d):
    self._d = d
    self._kernel_obj = c_void_p()
    lib.callsl("sl_create_kernel", "linear", d, byref(self._kernel_obj))
    
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

    if subtype is 'fast':
      return sketch.FastGaussianRFT(self._d, s, self._sigma, defouttype, **kargs)
    else:
      return sketch.GaussianRFT(self._d, s, self._sigma, defouttype, **kargs)


class Polynomial(Kernel):
  """
  A object representing the polynomial kernel over d dimensional vectors, with
  bandwidth exponent q and parameter c.

  Kernel function is :math:`k(x,y)=(\\gamma x^T y + c)^q`.

  :param d: dimension of vectors on which kernel operates.
  :param q: exponent of the kernel.
  :param c: kernel parameter, must be >= 0.
  """

  def __init__(self, d, q=3, c=0, gamma=1):
    if c < 0:
      raise ValueError("kernel paramter must be >= 0")
    if type(q) is not int:
      raise errors.InvalidParamterError("exponent must be integer")

    self._d = d
    self._q = q
    self._c = c
    self._gamma = gamma

    self._kernel_obj = c_void_p()
    lib.callsl("sl_create_kernel", "polynomial", self._d, byref(self._kernel_obj), \
                c_int(self._q), c_double(self._gamma), c_double(self._c))
    
  def rft(self, s, subtype=None, defouttype=None, **kargs):
    """
    Create a random features transform for the kernel.
    This function uses TensorSketch (Pahm-Pagh Transform)
    
    :param s: number of random features.
    :param subtype: subtype of rft to use (e.g. sparse, fast).
           Currently we support only regular (None), but we keep
           this argument to have a unifying interface.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """

    return sketch.PPT(self._d, s, self._q, self._c, self._gamma, defouttype, **kargs)


class Laplacian(Kernel):
  """
  A object representing the Laplacian kernel over d dimensional vectors, with
  bandwidth sigma.

  :param d: dimension of vectors on which kernel operates.
  :param sigma: bandwidth of the kernel.
  """

  def __init__(self, d, sigma):
    self._d = d
    self._sigma = sigma
    self._kernel_obj = c_void_p()
    lib.callsl("sl_create_kernel", "laplacian", d, byref(self._kernel_obj), c_double(sigma))
  
  def rft(self, s, subtype=None, defouttype=None, **kwargs):
    """
    Create a random features transform for the kernel.
    This function uses random Fourier features (Rahimi-Recht).
    
    :param s: number of random features.
    :param subtype: subtype of rft to use (e.g. sparse, fast).
           Currently we support regular (None) and quasirandom.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """
    if subtype is 'quasirandom':
      return sketch.LaplacianQRFT(self._d, s, self._sigma, defouttype, **kwargs)
    else:
      return sketch.LaplacianRFT(self._d, s, self._sigma, defouttype, **kwargs)


class ExpSemiGroup(Kernel):
  """
  A object representing the Exponential Semigroup kernel over d dimensional vectors, with
  parameter beta.

  :param d: dimension of vectors on which kernel operates.
  :param beta: kernel parameter
  """

  def __init__(self, d, beta):
    self._d = d
    self._beta = beta
    self._kernel_obj = c_void_p()
    lib.callsl("sl_create_kernel", "expsemigroup", d, byref(self._kernel_obj))
  
  def rft(self, s, subtype=None, defouttype=None, **args):
    """
    Create a random features transform for the kernel.
    This function uses random Laplace features.
    
    :param s: number of random features.
    :param subtype: subtype of rlt to use (e.g. sparse, fast).
           Currently we support only regular (None), but we keep
           this argument to have a unifying interface.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """

    return sketch.ExpSemigroupRLT(self._d, s, self._beta, defouttype, **args)


class Matern(Kernel):
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
    self._kernel_obj = c_void_p()
    lib.callsl("sl_create_kernel", "matern", d, byref(self._kernel_obj), \
              c_double(self_nu), c_double(self._l))

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