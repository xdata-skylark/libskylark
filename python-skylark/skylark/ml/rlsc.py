import skylark
from skylark import base
import skylark.lib as lib

from ctypes import byref, c_void_p, c_double, c_int

class RLSCParams(base.Params):
    """
    Parameter object for Krr.
    """

    def __init__(self):
        super(RLSCParams, self).__init__()
        self.use_fast = False
        self.sketched_rls = False
        self.sketch_size = -1
        self.fast_sketch = False
        self.tolerance = 1e-3
        self.res_print = 10
        self.iter_lim = 1000
        self.max_split = 0


def kernel_rlsc(X, L, A, k, lambda_, dir="columns", params=None):
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
    L = lib.adapt(L)
    A = lib.adapt(A)
    

    Xobj = X.ptr()
    Lobj = L.ptr()
    Aobj = A.ptr()

    if (Xobj == -1 or Lobj == -1 or Aobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as X, L or A")

    rcoding = c_void_p()

    # use default params in case none are provided
    if params == None:
        params = RLSCParams()
    params_json = params.str() + '\0'

    lib.callsl("sl_kernel_rlsc", \
            dir, k.get_obj(), \
            X.ctype(), Xobj, \
            L.ctype(), Lobj, \
            c_double(lambda_), \
            A.ctype(), Aobj, \
            rcoding, \
            params_json)

    X.ptrcleaner()
    L.ptrcleaner()
    A.ptrcleaner()

    return (A.getobj())