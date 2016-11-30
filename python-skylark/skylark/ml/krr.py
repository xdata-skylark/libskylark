import skylark
from skylark import base
import skylark.lib as lib

from ctypes import byref, c_void_p, c_double, c_int

class KrrParams(base.Params):
    """
    Parameter object for Krr.
    """

    def __init__(self):
        super(KrrParams, self).__init__()
        self.use_fast = False
        self.sketched_rr = False
        self.sketch_size = -1
        self.fast_sketch = False
        self.tolerance = 1e-3
        self.res_print = 10
        self.iter_lim = 1000
        self.max_split = 0


def KernelRidge(X, Y, A, k, lambda_, dir="columns", params=None):
    """
    TODO: Description
    """

    if dir == 0 or dir == "columns":
      dir = 1
    elif dir == 1 or dir == "rows":
      dir = 2
    else:
      raise ValueError("Direction must be either columns/rows or 0/1")


    X = lib.adapt(X)
    Y = lib.adapt(Y)
    A = lib.adapt(A)
    

    Xobj = X.ptr()
    Yobj = Y.ptr()
    Aobj = A.ptr()

    if (Xobj == -1 or Yobj == -1 or Aobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as X, Y or A ")

    # use default params in case none are provided
    if params == None:
        params = KrrParams()
    params_json = params.str() + '\0'

    lib.callsl("sl_kernel_ridge", \
            dir, k.get_obj(), \
            X.ctype(), Xobj, \
            Y.ctype(), Yobj, \
            c_double(lambda_), \
            A.ctype(), Aobj, \
            params_json)

    X.ptrcleaner()
    Y.ptrcleaner()
    A.ptrcleaner()

    return (A.getobj())

def ApproximateKernelRidge(X, Y, A, k, lambda_, s, dir="columns", params=None):
    """
    TODO: Description
    """

    if dir == 0 or dir == "columns":
      dir = 1
    elif dir == 1 or dir == "rows":
      dir = 2
    else:
      raise ValueError("Direction must be either columns/rows or 0/1")


    X = lib.adapt(X)
    Y = lib.adapt(Y)
    A = lib.adapt(A)
    

    Xobj = X.ptr()
    Yobj = Y.ptr()
    Aobj = A.ptr()

    if (Xobj == -1 or Yobj == -1 or Aobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as X, Y or A ")

    S = c_void_p()

    # use default params in case none are provided
    if params == None:
        params = KrrParams()
    params_json = params.str() + '\0'

    lib.callsl("sl_approximate_kernel_ridge", \
            dir, k.get_obj(), \
            X.ctype(), Xobj, \
            Y.ctype(), Yobj, \
            c_double(lambda_), \
            S, \
            A.ctype(), Aobj, \
            c_int(s), \
            lib.ctxt_obj, \
            params_json)

    X.ptrcleaner()
    Y.ptrcleaner()
    A.ptrcleaner()

    return (A.getobj())


def FasterKernelRidge(X, Y, A, k, lambda_, s, dir="columns", params=None):
    """
    TODO: Description
    """

    if dir == 0 or dir == "columns":
      dir = 1
    elif dir == 1 or dir == "rows":
      dir = 2
    else:
      raise ValueError("Direction must be either columns/rows or 0/1")


    X = lib.adapt(X)
    Y = lib.adapt(Y)
    A = lib.adapt(A)
    

    Xobj = X.ptr()
    Yobj = Y.ptr()
    Aobj = A.ptr()

    if (Xobj == -1 or Yobj == -1 or Aobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as X, Y or A ")

    # use default params in case none are provided
    if params == None:
        params = KrrParams()
    params_json = params.str() + '\0'

    lib.callsl("sl_faster_kernel_ridge", \
            dir, k.get_obj(), \
            X.ctype(), Xobj, \
            Y.ctype(), Yobj, \
            c_double(lambda_), \
            A.ctype(), Aobj, \
            c_int(s), \
            lib.ctxt_obj,
            params_json)

    X.ptrcleaner()
    Y.ptrcleaner()
    A.ptrcleaner()

    return (A.getobj())