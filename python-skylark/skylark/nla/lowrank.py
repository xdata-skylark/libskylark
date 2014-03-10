"""
Various low-rank approximations.
"""

import numpy

def approximate_domsubspace_basis(A, k, s, t, 
                                  kernel=None, subtype=None):
    """
    For a matrix Z with k columns that approximates the k-dominant subspace
    spanned by A (or, described by A and a kernel).

    If :math:`s=\\Omega (k / \\epsilon)` and  :math:`t=\\Omega (k / \\epsilon^2)`
    we are guarenteed to find a Z such that 
       :math:`\\|Z Z^T A - A\\|_F \\leq (1 +\\epsilon)\|A_k - A\|_F`

    Except Z, return also S, R, V where S is a sketching transform and R
    and V are matrices such that :math:`Z=S(A)R^{-1}V`.

    Using kernels: you can optionally supply a kernel, in which case
    A in the above description is replaced by :math:`\\phi(A)` where
    :math:`\phi` is the implicit mapping defined by the kernel.

    :param A: input matrix
    :param k: target rank
    :param s,t: sketching sizes.
    :param kernel - kernel to use
    :param subtype - subtype of random features transform for the kernel to use.

    :returns: Z, S, R, V
    
    """
    # TODO: make this work for all matrix types.
    d = A.shape[1]
    if kernel is None:
        from skylark.ml import kernels
        kernel = kernels.Linear(d)
    S = kernel.rft(s, subtype)
    X = S / A
    T = kernel.rft(t, subtype)
    Y = T / A
    U, R = numpy.linalg.qr(X)
    M, s, N = numpy.linalg.svd(numpy.dot(U.T, Y), 0)
    V = M[:, 0:k]
    Z = numpy.dot(U, V)
    return Z, S, R, V




