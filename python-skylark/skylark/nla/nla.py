__all__ = ['SVDParams', 'FasterLeastSquaresParams', 'approximate_svd', 'approximate_symmetric_svd', 'faster_least_squares']
from skylark import base
from skylark import errors
from skylark import lib
import El

class SVDParams(base.Params):
    """
    Parameter object for SVD.
    """

    def __init__(self):
        super(SVDParams, self).__init__()
        self.oversampling_ratio = 2
        self.oversampling_additive = 0
        self.num_iterations = 2
        self.skip_qr = False

class FasterLeastSquaresParams(base.Params):
    """ 
    Parameter object for faster least squares.
    """

    def __init__(self):
        super(FasterLeastSquaresParams, self).__init__()


def get_params(params_type, params):
    if params == None:
        if params_type == "SVDParams":
            params = SVDParams()
        elif params_type == "FasterLeastSquaresParams":
            params = FasterLeastSquaresParams()

    return params.str() + '\0'

def is_valid(matrices):
    for m in matrices:
        if m.ptr() == -1:
            raise errors.InvalidObjectError("Invalid/unsupported object passed to the adaptor")

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

    A, U, S, V = lib.adapt([A, U, S, V])

    validate([A, U, S, V])

    params_json = get_params("SVDParams", params)

    lib.callsl("sl_approximate_svd", \
            A.ctype(), A.ptr(), \
            U.ctype(), U.ptr(), \
            S.ctype(), S.ptr(), \
            V.ctype(), V.ptr(), \
            k, params_json, lib.ctxt_obj)

    lib.clean_pointer([A, U, S, V])

    return (U.getobj(), S.getobj(), V.getobj())

def approximate_symmetric_svd(A, S, V, k=10, params=None):
    """
    Compute the SVD of symmetric **A** such that **SVD(A) = V S V^T**.

    :param A: Input matrix.
    :param S: Output S (singular values).
    :param V: Output V (right singular vectors).
    :param k: Dimension to apply along.
    :param params: Parmaters for the SVD.
    :returns: (S, V)
    """

    A, S, V = lib.adapt([A, S, V])

    validate([A, S, V])

    # use default params in case none are provided
    params_json = get_params("SVDParams", params)

    lib.callsl("sl_approximate_symmetric_svd", \
            A.ctype(), A.ptr(), \
            S.ctype(), S.ptr(), \
            V.ctype(), V.ptr(), \
            k, params_json, lib.ctxt_obj)

    lib.clean_pointer([A, S, V])

    return (S.getobj(), V.getobj())

def faster_least_squares(A, B, X, orientation=El.NORMAL, params=None):
    """
    Compute a solution to the least squares problem:

    If orientation == El.NORMAL: argmin_X ||A * X - B||_F
    If orientation == El.ADJOINT: argmin_X ||A^H * X - B||_F

    * ADJOINT not yet supported! *

    :param A: Input matrix.
    :param B: Right hand side.
    :param X: Solution
    :param params: Parmaters.
    :returns: X
    """
    
    A, B, X = lib.adapt([A, B, X])

    validate([A, B, X])

    # use default params in case none are provided
    params_json = get_params("FasterLeastSquaresParams", params)

    lib.callsl("sl_faster_least_squares", \
               orientation, \
               A.ctype(), A.ptr(), \
               B.ctype(), B.ptr(), \
               X.ctype(), X.ptr(), \
               params_json, lib.ctxt_obj)

    lib.clean_pointer([A, B, X])
  
    return X.getobj()
