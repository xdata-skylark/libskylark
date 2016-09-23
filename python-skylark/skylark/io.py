__all__ = ['readlibsvm']

from skylark import base
from skylark import errors
from skylark import lib

def readlibsvm(fname, X, Y, direction="rows", min_d=0, max_n=-1):

    cdirection = None
    if direction == 0 or direction == "columns":
      cdirection = 1
    if direction == 1 or direction == "rows":
      cdirection = 2
    if cdirection is None:
      raise ValueError("Direction must be either columns/rows or 0/1")
  
    X = lib.adapt(X)
    Y = lib.adapt(Y)

    Xobj = X.ptr()
    Yobj = Y.ptr()

    if (Xobj == -1 or Yobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as X, Y")

    lib.callsl("sl_readlibsvm", fname, \
               X.ctype(), Xobj, \
               Y.ctype(), Yobj, \
               cdirection, min_d, max_n)

    X.ptrcleaner()
    Y.ptrcleaner()
 
    return (X.getobj(), Y.getobj())
