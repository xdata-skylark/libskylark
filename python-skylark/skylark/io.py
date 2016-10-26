__all__ = ['readlibsvm']

from skylark import base
from skylark import errors
from skylark import lib
import skylark.utils as utils


def readlibsvm(fname, X, Y, direction="rows", min_d=0, max_n=-1):

    cdirection = utils.get_direction(direction)
  
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
