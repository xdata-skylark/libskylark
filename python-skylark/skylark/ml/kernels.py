import numpy 
import skylark
from ctypes import byref, c_void_p, c_double, c_int
from skylark import sketch, errors
from distances import euclidean
import scipy.special
import scipy
import sys, math
import skylark.lib as lib

def __get_direction(dir):
  if dirX == 0 or dirX == "columns":
    return 1
  elif dirX == 1 or dirX == "rows":
    return 2
  else:
    raise ValueError("Direction must be either columns/rows or 0/1")

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
  else:
    raise ValueError("kerneltype not recognized")

def __gram(self, X, K, kernel_ptr, dirX="rows", dirY="rows", Y=None):
    """
    Given a kernel, returns the dense Gram matrix evaluated over the datapoints.
  
    :param X: n-by-d data matrix
    :param K: placeholder for output Gram matrix.
    :param kernel_ptr: kernel to be executed
    :param Y: another data matrix. If Y is None, then X is used.
    """

    if Y is None:
        Y = X
        dirY = dirX
    
    cdirX = get_direction(dirX)
    cdirY = get_direction(dirY)

    X = lib.adapt(X)
    Y = lib.adapt(Y)
    K = lib.adapt(K)

    lib.callsl("sl_kernel_gram", cdirX, cdirY, _kernel_obj, \
               X.ctype(), X.ptr(), \
               Y.ctype(), Y.ptr(), \
               K.ctype(), K.ptr())

    X.ptrcleaner()
    Y.ptrcleaner()
    K.ptrcleaner()

    return K.getobj()
 