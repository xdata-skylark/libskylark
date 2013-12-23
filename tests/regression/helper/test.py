import numpy as np

def test_helper(A, M, N, R, sketch, measures, MPI, num_repeats=5,
                intype="DistMatrix_VR_STAR", direction="columnwise"):
    """
    Test if the singular values of the original (M x N) and sketched matrix
    (R x N) are fulfilling some measurement criteria. The test is repeated
    num_repeats times.
    """
    results = []
    for i in range(num_repeats):

        if direction == "columnwise":
            S  = sketch(M, R, intype=intype)
            SA = np.zeros((R, N), order='F')
        else:
            S  = sketch(N, R, intype=intype)
            SA = np.zeros((M, R), order='F')

        S.apply(A, SA, direction)

        SA = MPI.COMM_WORLD.bcast(SA, root=0)

        for m in measures:
            results.append(m(SA))

    return results
