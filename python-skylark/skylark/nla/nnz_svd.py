def nnz_svd(A, k, s1, s3, s2=0, s4=0, debug=0):
    """
    Input-Sparsity Time Low-Rank Approximation

    We are mapping here the algorithm specified under Theorem 47 in Clarkson &
    Woodruff, Low-Rank Approximation and Regression in Input Sparsity Time.

    Assumptions made for this implementation:
    (1) n, d are very large and nnz(A) <<< n*d.
    (2) k is small;  (n x k) and (d x k) fit in single node memory
    (3) (n x s2), (s4 x d), and (s2 x d) fit in single node memory

    Notation:
    (1) columwise sketching reduces the dimension of the columns; that is, the
        number of rows are reduced. We also think of this as multiplication of
        A from left.
    (2) rowwise sketching reduces the dimension of the rows; that is, the
        number of columns are reduced. We also think of this as multiplication
        of A from the right.

    :param A: is an n x d matrix with nnz(A) non-zero entries from KDT.
    :param k: is the desired rank?
    :param s1: sketching parameter for CWT rowwise sketching
    :param s3: sketching parameter for CWT columnwise sketching
    :param s2: (optional) sketching parameter for FJLT rowwise sketching
    :param s4: (optional) sketching parameter for FJLT columnwise sketching
    :param debug: (optional) parameter for the level of debugging needed

    :returns: L_lcl, D_lcl, W_lcl, such that L*D*W is a rank-k
              approximation to A.
    """

    # prevent mpi4py from calling MPI_Finalize()
    import mpi4py.rc
    mpi4py.rc.finalize = False

    from mpi4py import MPI
    from skylark import sketch

    import elem, kdt, math, numpy

    #TODO: Check that the type is a kdt matrix

    # 1. Ensure that all the dimensions are correct:
    n   = A.nrow()
    d   = A.ncol()
    nnz = A.nnn()

    # In case nothing was passed for s2, set it to be s1 shorting the FJLT
    if 0 >= s2:
        s2 = s1

    # In case nothing was passed for s4, set it to be s3 shorting the FJLT
    if 0 >= s4:
        s4 = s3

    try:
        d >= s1 and s1 >= s2
    except:
        print "We require that (d >= s1) and (s1 >= s2)"
        return

    try:
        n >= s3 and s3 >= s4
    except:
        print "We require that (n >= s3) and (s3 >= s4)"
        return

    try:
        k <= s2 and k <= s4
    except:
        print "We require that (k <= s2) and (k <= s4)"
        return

    if debug>=1:
        print '--------------------------------------------------------'
        print 'Starting nnz svd'
        print 'Sizes:: n=%d, d=%d, k=%d, s1=%d, s2=%d, s3=%d, s4=%d' % \
               (n, d, k, s1, s2, s3, s4)
        print '--------------------------------------------------------'

    # Start of the algorithm
    #########################################################################

    # Reduce A from (n,d) to (n,s1) using CWT
    # Uncomment this line when you want to use distributed matrices
    # A1 = elem.DistMatrix_d_VC_STAR(n, s1);
    A1 = numpy.zeros((n, s1), order='F')
    CWT_d_to_s1 = sketch.CWT(d, s1)
    CWT_d_to_s1.apply(A, A1, "rowwise")

    if (debug==2):
        print 'A1 shape = (%d,%d)' % A1.shape

    # If needed, further reduce (n,s1) to (n,s2) using FJLT
    if s2 == s1:
        A1_aka_U_lcl = A1;
    else:
        A1_aka_U_lcl = numpy.zeros((n, s2), order='F')
        FJLT_s1_to_s2 = sketch.FJLT(s1, s2)
        FJLT_s1_to_s2.apply(A1, A1_aka_U_lcl, "rowwise")

    # Get the QR (U,R=QR(A)) decomposition to get orthogonal basis U
    elem.qr_Explicit(A1_aka_U_lcl);

    if debug==2:
        print 'U shape = (%d,%d)' % A1_aka_U_lcl.shape

    # Reduce the column dimension of U from (n,s2) to (s3,s2)
    SU_1_lcl = numpy.zeros((s3, s2), order='F')
    CWT_n_to_s3 = sketch.CWT(n, s3)
    CWT_n_to_s3.apply(A1_aka_U_lcl, SU_1_lcl, "columnwise")

    if debug==2:
        print 'SU shape = (%d,%d)' % SU_1_lcl.shape

    # If needed, further reduce the column dimension from (s3,s2) to (s4,s2)
    if s4 == s3:
        SU_2_lcl = SU_1_lcl
    else:
        SU_2_lcl = numpy.zeros((s4, s2), order='F')
        FJLT_s3_to_s4 = sketch.FJLT(s3, s4)
        FJLT_s3_to_s4.apply(SU_1_lcl, SU_2_lcl, "columnwise")

    # Compute the SVD of SU
    Ut_lcl, St_lcl, Vt_lcl = smart_svd (SU_2_lcl)

    if debug==2:
        print 'Ut shape = (%d,%d)' % Ut_lcl.shape
        print 'St shape = (%d,%d)' % numpy.diag(St_lcl).shape
        print 'Vt shape = (%d,%d)' % Vt_lcl.shape

    # Reduce the column dimension of A from (n,d) to (s3,d)
    # Uncomment this line when you want to use distributed matrices
    #SA = elem.DistMatrix_d_VC_STAR(s3, d);
    SA = numpy.zeros((s3, d), order='F')
    CWT_n_to_s3.apply(A, SA, "columnwise")

    if debug==2:
        print 'SA shape = (%d,%d)' % SA.shape

    # If needed, further reduce the column dimension from (s3,d) to (s4,d)
    if s4 == s3:
        SA_lcl = SA
    else:
        SA_lcl = numpy.zeros((s4, d), order='F')
        FJLT_s3_to_s4.apply(SA, SA_lcl, "columnwise")

    # Z = Ut'*SA
    Z_lcl = numpy.zeros((Ut_lcl.shape[1], SA_lcl.shape[1]), order='F')
    elem.Gemm(elem.TRANSPOSE, elem.NORMAL, 1.0, Ut_lcl, SA_lcl, 0.0, Z_lcl)

    if debug==2:
        print 'Z shape = (%d,%d)' % Z_lcl.shape

    # Ub,Sb,Vb = svd(Z, k)
    Ub_lcl, Sb_lcl, Vb_lcl = smart_svd (Z_lcl)

    if debug==2:
        print 'Ub shape = (%d,%d)' % Ub_lcl.shape
        print 'Sb shape = (%d,%d)' % numpy.diag(Sb_lcl).shape
        print 'Vb shape = (%d,%d)' % Vb_lcl.shape

    # Truncate to rank-k
    Ub_aka_Y_k_lcl = Ub_lcl[:,0:k]
    Sb_k_lcl = Sb_lcl[0:k]
    #Vb_k_lcl = Vb_lcl[0:k,:]
    Vb_k_lcl = Vb_lcl[:,0:k]

    if debug==2:
        print 'Truncated Ub shape = (%d,%d)' % Ub_aka_Y_k_lcl.shape
        print 'Truncated Sb shape = (%d,%d)' % numpy.diag(Sb_k_lcl).shape
        print 'Truncated Vb shape = (%d,%d)' % Vb_k_lcl.shape

    # Take pseudo-inverse of St by replacing its diagonal by 1/diagonal (zeros
    # on the diagonal will be left untouched).
    nz_ele = numpy.nonzero(St_lcl)
    St_lcl[nz_ele] = 1. / St_lcl[nz_ele]

    # Computing St^{-} * Ub_k * Sb_k
    elem.DiagonalScale(elem.LEFT,  elem.NORMAL, St_lcl,   Ub_aka_Y_k_lcl)
    elem.DiagonalScale(elem.RIGHT, elem.NORMAL, Sb_k_lcl, Ub_aka_Y_k_lcl)

    if debug==2:
        print 'Y shape = (%d,%d)' % Ub_aka_Y_k_lcl.shape

    # Uh, D, Vh = svd(Y)
    Uh_lcl, D_lcl, Vh_lcl = smart_svd (Ub_aka_Y_k_lcl)

    if debug==2:
        print 'Uh shape = (%d,%d)' % Uh_lcl.shape
        print 'D shape  = (%d,%d)' % numpy.diag(D_lcl).shape
        print 'Vh shape = (%d,%d)' % Vh_lcl.shape

    # L1 = Vt * Uh
    L1_lcl = numpy.zeros((Vt_lcl.shape[0], Uh_lcl.shape[1]), order='F')
    elem.Gemm(elem.NORMAL, elem.NORMAL, 1.0, Vt_lcl, Uh_lcl, 0.0, L1_lcl)

    if debug==2:
        print 'L1 shape = (%d,%d)' % L1_lcl.shape

    # L = U * L1
    L_lcl = numpy.zeros((A1_aka_U_lcl.shape[0], L1_lcl.shape[1]), order='F')
    elem.Gemm(elem.NORMAL, elem.NORMAL, 1.0, A1_aka_U_lcl, L1_lcl, 0.0, L_lcl)

    if debug==2:
        print 'L shape = (%d,%d)' % L_lcl.shape

    # W = Vb_k * Vh;
    W_lcl = numpy.zeros((Vb_k_lcl.shape[0], Vh_lcl.shape[1]), order='F')
    elem.Gemm(elem.NORMAL, elem.NORMAL, 1.0, Vb_k_lcl, Vh_lcl, 0.0, W_lcl)

    if debug==2:
        print 'W shape = (%d,%d)' % W_lcl.shape

    if debug>=1:
        print '------------------------------------------------------'
        print 'Returning L=(%d,%d)' % L_lcl.shape
        print '          D=(%d,%d)' % numpy.diag(D_lcl).shape
        print '          W=(%d,%d)' % W_lcl.T.shape
        print '------------------------------------------------------'

    # Ken's paper says that A = L*D*W, so return W.T
    return L_lcl, D_lcl, W_lcl.T


def smart_svd(A_lcl):
    """
    Smart SVD that checks if the matrix is tall-skinny or short-fat and
    creates the sizes of U, S, V appropriately. Although it currently turns
    around and calls numpy, it serves an important function of abstraction.
    This can be exploited later when we need to reintroduce Elemental matrices

    :param: A, an (nxd) numpy local dense matrix
    :returns: U, S, V, the singular value decomposition of the matrices

    Caveat:
    (1) Elemental resizes the matrices as it sees fit, so S becomes
        a column vector (which represents a diagonal matrix; saves space)

    Note: This preserves A and does not overwrite it
    """

    import numpy

    # Get the sizes from the matrices
    n, d = A_lcl.shape

    if (n >= d):
        # In this case, U=(nxd), S=(dxd), and V=(dxd)
        U_lcl, S_lcl, V_lcl = numpy.linalg.svd(A_lcl, False);

        return U_lcl, S_lcl, V_lcl
    else:
        # In this case, U=(nxn), S=(nxn), and V=(dxn)
        U_lcl, S_lcl, V_lcl = numpy.linalg.svd(A_lcl, False);

        # Notice that numpy returns V, when it really returns V.T; Elemental
        # is not this way. To clear any confusion, we will use the standard
        # decomposition of A = U*S*V.T. So, we return the transpose
        return U_lcl, S_lcl, V_lcl.T

