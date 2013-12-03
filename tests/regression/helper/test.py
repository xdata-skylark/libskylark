import numpy as np
from collections import namedtuple

def svd_bound(A, SA):
    """
    Test if the singular values of the original (M x N) and sketched matrix
    (R x N) are bounded by:

        SVD(A)_i * (1 - accuracy) <= SVD(SA)_i <= SVD(A)_i * (1 + accuracy)

    Computes the average relative error per index and additionally we returns
    a boolean vector describing, for each index, if we have at leaste one
    singular value honoring the bounds.
    """

    result = namedtuple('svd_bound_result', ['average', 'success'])

    #FIXME: A.Matrix will not work in parallel
    sv  = np.linalg.svd(A.Matrix, full_matrices=1, compute_uv=0)
    sav = np.linalg.svd(SA,       full_matrices=1, compute_uv=0)

    average = abs(sv - sav) / sv
    success = np.zeros(len(sv))

    for idx in range(len(sv)):
        success[idx]  = success[idx] or (sv[idx] * (1 - accuracy) <= sav[idx] <= sv[idx] * (1 + accuracy))

    return result._make([average, success])


def test_helper(A, M, N, R, sketch, measures,
                accuracy=0.5, num_repeats=5,
                intype="DistMatrix_VR_STAR", direction="columnwise"):
    """
    Test if the singular values of the original (M x N) and sketched matrix
    (R x N) are fullfilling some measurement criteria.
    The test is repeated num_repeats times.
    """
    results = []
    for i in range(num_repeats):
        S  = sketch(M, R, intype=intype)
        SA = np.zeros((R, N), order='F')
        S.apply(A, SA, direction)
        SA = MPI.COMM_WORLD.bcast(SA, root=0)

        for m in measures:
            results.append(m(A, SA))

    return results
