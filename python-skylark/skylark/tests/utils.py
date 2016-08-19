import El
import numpy as np

def to_num_py(M):
    if isinstance(M, El.Matrix):
        return M.ToNumPy()
    elif (isinstance(M, El.DistMatrix)):
        return M.Matrix().ToNumPy()
    return M

def substract(A, B, m_type=El.Matrix):
    return A - B

def equal(A, B, threshold=1.e-4):
    A = to_num_py(A)
    B = to_num_py(B)
    return (substract(A, B) < threshold).all()