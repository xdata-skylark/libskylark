import numpy 
import skylark
from skylark import sketch
from distances import euclidean
import sys

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
  
  def rft(self, s, defouttype=None):
    """
    Create a random features transform for the kernel.
    This function uses random Fourier features (Rahimi-Recht).
    
    :param s: number of random features.
    :param defouttype: default output type for the transform.
    :returns: random features sketching transform object.
    """
    
    return sketch.GaussianRFT(self._d, s, self._sigma, defouttype)



        
