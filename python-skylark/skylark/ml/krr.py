import skylark
from skylark import base
import skylark.lib as lib
import skylark.utils as utils

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

    dir = utils.get_direction(dir)

    X = utils.adapt_and_check(X)
    Y = utils.adapt_and_check(Y)
    A = utils.adapt_and_check(A)
    

    # use default params in case none are provided
    if params == None:
        params = KrrParams()
    params_json = params.str() + '\0'

    lib.callsl("sl_kernel_ridge", \
            dir, k.get_kernel_obj(), \
            X.ctype(), X.ptr(), \
            Y.ctype(), Y.ptr(), \
            c_double(lambda_), \
            A.ctype(), A.ptr(), \
            params_json)

    X.ptrcleaner()
    Y.ptrcleaner()
    A.ptrcleaner()

    return (A.getobj())