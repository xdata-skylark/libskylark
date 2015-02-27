import El

def test_helper(A, M, N, R, sketch, measures, MPI, num_repeats=5, direction="columnwise"):
    """
    Test if the singular values of the original (M x N) and sketched matrix
    (R x N) are fulfilling some measurement criteria. The test is repeated
    num_repeats times.
    """
    results = []
    for i in range(num_repeats):

        if direction == "columnwise":
            S  = sketch(M, R)
            SA = El.DistMatrix(El.dTag, El.STAR, El.STAR)
            El.Uniform(SA, R, N)
        else:
            S  = sketch(N, R)
            SA = El.DistMatrix(El.dTag, El.STAR, El.STAR)
            El.Uniform(SA, M, R)

        S.apply(A, SA, direction)

        for m in measures:
            results.append(m(SA.Matrix().ToNumPy()))

    return results
