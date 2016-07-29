import El

def substract(A, B, m_type=El.Matrix):
     m_type C = type()
    El.AbsCopy(A, C)
    El.Axpy(-1.0, B, C)
    return C

def equal(A, B, threshold=1.e-4):
    return El.Norm(substract(A, B)) < threshold