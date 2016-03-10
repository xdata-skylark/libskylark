from skylark import errors
from skylark import sketch as sk
from ctypes import byref, c_void_p

import json

class Params():
    """
    Helper object capturing parameters for the randomized SVD implementation.
    """

    def __init__(self):
        self.oversampling_ratio = 2
        self.oversampling_additive = 0
        self.num_iterations = 0
        self.skip_qr = False
        self.am_i_printing = False
        self.log_level = 0
        self.prefix = ""
        self.debug_level = 0

    def str(self):
        return json.dumps(self, default=lambda obj: obj.__dict__,
                sort_keys=True)


def approximate_svd(A, k=10, parms=None, resultType=None):
    """
    Compute the SVD of **A** such that **SVD(A) = U S V^T**.

    :param A: Input matrix.
    :param k: Dimension to apply along.
    :param parms: Parmaters for the SVD.
    :param resultType: Type of U, S, V matrices.
    :returns: (U, S, V)
    """

    A = sk._adapt(A)
    if resultType == None:
        resultType = A
    else:
        resultType = sk._adapt(resultType)
    ctor = resultType.getctor()

    U = ctor(0, 0, resultType)
    S = ctor(0, 0, resultType)
    V = ctor(0, 0, resultType)

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
    if parms == None:
        parms = Params()
        parms.num_iterations = 2
    parms_json = parms.str() + '\0'

    sk._callsl(sk._lib.sl_approximate_svd, \
            A.ctype(), Aobj, \
            U.ctype(), Uobj, \
            S.ctype(), Sobj, \
            V.ctype(), Vobj, \
            k, parms_json, sk._ctxt_obj)

    A.ptrcleaner()
    U.ptrcleaner()
    S.ptrcleaner()
    V.ptrcleaner()

    return (U.getobj(), S.getobj(), V.getobj())

