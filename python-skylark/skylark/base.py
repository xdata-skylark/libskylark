import json
import skylark.lib as lib

from ctypes import byref, c_void_p, c_double, c_int

class Params(object):
    """
    Base parameter object
    """
    def __init__(self):
        self.am_i_printing = False
        self.log_level = 0
        self.prefix = ""
        self.debug_level = 0

    def str(self):
        return json.dumps(self, default=lambda obj: obj.__dict__,
                sort_keys=True)

def gaussian_matrix(A, m, n):
    """
    TODO: Description
    """

    A = lib.adapt(A)
    Aobj = A.ptr()

    if (Aobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as A ")

    lib.callsl("sl_gaussian_matrix",  \
            A.ctype(), Aobj, \
            c_int(m), c_int(n), \
            lib.ctxt_obj)

    A.ptrcleaner()

    return (A.getobj())

def uniform_matrix(A, m, n):
    """
    TODO: Description
    """

    A = lib.adapt(A)
    Aobj = A.ptr()

    if (Aobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as A ")

    lib.callsl("sl_uniform_matrix",  \
            A.ctype(), Aobj, \
            c_int(m), c_int(n), \
            lib.ctxt_obj)

    A.ptrcleaner()

    return (A.getobj())
