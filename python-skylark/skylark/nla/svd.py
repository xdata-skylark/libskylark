from skylark import errors
from skylark import sketch as sk
from ctypes import byref, c_void_p

import json

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
    
class SVDParams(Params):
    """
    Parameter object for SVD.
    """

    def __init__(self):
        super(SVDParams, self).__init__()
        self.oversampling_ratio = 2
        self.oversampling_additive = 0
        self.num_iterations = 2
        self.skip_qr = False

def approximate_svd(A, U, S, V, k=10, params=None):
    """
    Compute the SVD of **A** such that **SVD(A) = U S V^T**.

    :param A: Input matrix.
    :param U: Output U (left singular vectors).
    :param S: Output S (singular values).
    :param V: Output V (right singular vectors).
    :param k: Dimension to apply along.
    :param params: Parmaters for the SVD.
    :returns: (U, S, V)
    """

    
    A = sk._adapt(A)
    U = sk._adapt(U)
    S = sk._adapt(S)
    V = sk._adapt(V)

    Aobj = A.ptr()
    Uobj = U.ptr()
    Sobj = S.ptr()
    Vobj = V.ptr()

    if (Aobj == -1 or Uobj == -1 or Sobj == -1 or Vobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as A, U, S or V ")

    # use default params in case none are provided
    if params == None:
        params = SVDParams()
    params_json = params.str() + '\0'

    sk._callsl(sk._lib.sl_approximate_svd, \
            A.ctype(), Aobj, \
            U.ctype(), Uobj, \
            S.ctype(), Sobj, \
            V.ctype(), Vobj, \
            k, params_json, sk._ctxt_obj)

    A.ptrcleaner()
    U.ptrcleaner()
    S.ptrcleaner()
    V.ptrcleaner()

    return (U.getobj(), S.getobj(), V.getobj())

