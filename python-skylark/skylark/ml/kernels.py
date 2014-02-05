import numpy 
import skylark
from skylark import sketch, errors
from distances import euclidean
import sys

def kernel(kerneltype, d, **params):
  """
  Returns a kernel based on the input parameters.

  :param kerneltype: string identifying the kernel requested.
  :param d: dimension of the kernel.
  :param params: dictonary of kernel parameters, kernel dependent.
  :returns: kernel object
  """
  if not isinstance(kerneltype, str):
    raise ValueError("kerneltype must be a string")
  elif kerneltype.lower() == "linear":
    return Linear(d, **params)
  elif kerneltype.lower() == "gaussian":
    return Gaussian(d, **params)
  elif kerneltype.lower() == "polynomial":
    return Polynomial(d, **params)
  else:
    raise ValueError("kerneltype not recognized")

class Linear(object):
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
  
  def rft(self, s, subtype=None, defouttype=None, **args):
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
      return sketch.JLT(self._d, s, defouttype, **args)
    elif subtype is 'fast':
      return sketch.FJLT(self._d, s, defouttype, **args)
    elif subtype is 'hash':
      return skethc.CWT(self._d, s, defouttype, **args)
    else:
      raise ValueError("invalide subtype supplied")

class Gaussian(object):
  """
  A object representing the Gaussian kernel over d dimensional vectors, with
  bandwidth sigma.

  :param d: dimension of vectors on which kernel operates.
  :param sigma: bandwidth of the kernel.
  """

  def __init__(self, d, sigma):
    self._d = d
    self._sigma = sigma
    
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

    sigma = self._sigma
    if Xt is None:
      K = numpy.exp(-euclidean(X, X)/(2*sigma**2))
    else:
      if Xt.shape[1] != self._d:
        raise ValueError("Xt must have vectors of dimension d")
      K = numpy.exp(-euclidean(X, Xt)/(2*sigma**2))
      
    return K
  
  def rft(self, s, subtype=None, defouttype=None, **args):
    """
    Create a random features transform for the kernel.
    This function uses random Fourier features (Rahimi-Recht).
    
    :param s: number of random features.
    :param subtype: subtype of rft to use (e.g. sparse, fast).
           Currently we support only regular (None), but we keep
           this argument to have a unifying interface.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """
    if subtype is 'fast':
      return sketch.FastGaussianRFT(self._d, s, self._sigma, defouttype, **args)
    else:
      return sketch.GaussianRFT(self._d, s, self._sigma, defouttype, **args)

class Polynomial(object):
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
      n = X.shape[0]
      K = numpy.power(numpy.dot(X, X.T) + self._c * numpy.ones((n,n)), self._q)
    else:
      if Xt.shape[1] != self._d:
        raise ValueError("Xt must have vectors of dimension d")
      n = X.shape[0]
      t = Xt.shape[0]
      K = numpy.power(numpy.dot(Xt, X.T) + self._c * numpy.ones((t,n)), self._q)
      
    return K
  
  def rft(self, s, subtype=None, defouttype=None, **args):
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
    
    return sketch.PPT(self._d, s, self._q, self._c, self._gamma, defouttype, **args)
        
