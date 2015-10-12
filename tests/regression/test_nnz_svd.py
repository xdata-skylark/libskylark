# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize = False

import numpy
import scipy
import scipy.sparse.linalg

import kdt
import elem
import skylark.nla.nnz_svd, skylark.sketch

from mpi4py import MPI


from skylark.nla.nnz_svd import nnz_svd


def kdt_matrix(m, n, rank):
    '''
    Returns a m x n kdt matrix of rank rank.

    This is designed to be a rank-deficient sparse matrix with
    rank singular values: 1,2,..,rank.
    '''

    max_rank   = min(m, n)

    try:
        rank < max_rank
    except:
        print 'rank should be smaller than height, width for a rank-deficient matrix'
        return

    # coordinate triplets:
    # outer product of row and column sublists from partitioning
    # the shuffled row and column ranges into rank parts
    # for successive outer products the singular values go as 1,2,...,rank
    r = m / rank
    c = n / rank
    row_range = numpy.array(range(m))
    col_range  = numpy.array(range(n))
    numpy.random.shuffle(row_range)
    numpy.random.shuffle(col_range)

    triplets = []
    for i in range(rank-1):
        singular_value = float(i + 1)
        rows = row_range[i * r : (i+1) * r]
        cols = col_range[i * c : (i+1) * c]
        val  = 1. / numpy.sqrt(r * c) * singular_value
        triplets.extend([row, col, val] for row in rows for col in cols)
    singular_value = float(rank)
    rows = row_range[(rank-1) * r :]
    cols = col_range[(rank-1) * c :]
    val  = 1. / numpy.sqrt((m - (rank-1) * r) * (n - (rank-1) * c)) * singular_value
    triplets.extend([row, col, val] for row in rows for col in cols)
    rows, cols, vals = zip(*triplets)
    nnz = len(triplets)

    icoords = kdt.Vec(nnz, sparse=False)
    jcoords = kdt.Vec(nnz, sparse=False)
    values  = kdt.Vec(nnz, sparse=False)
    for i in range(0, nnz):
        icoords[i] = float(rows[i])
        jcoords[i] = float(cols[i])
        values[i]  = float(vals[i])

    A = kdt.Mat(icoords, jcoords, values, n, m)
    return A

def kdt_to_numpy_dense(A):
    '''
    Converts a kdt matrix into a numpy ndarray
    '''

    m = A.nrow()
    n = A.ncol()
    rows, cols, vals = A.toVec()
    nnz = len(rows)
    B = numpy.zeros((m, n))
    for row, col, val in zip(rows, cols, vals):
        B[row, col] = val
    return B


def kdt_to_scipy_sparse(A):
    '''
    Convert a kdt matrix into a scipy.sparse coo_matrix
    '''

    m = A.nrow()
    n = A.ncol()
    rows, cols, vals = A.toVec()
    nnz = len(rows)
    B = scipy.sparse.coo_matrix((vals, (rows, cols)), shape = (m, n))
    return B

# dummy
def nnz_svd_dummy(A, k, s1=0, s2=0, s3=0, s4=0):
    A_dense = kdt_to_numpy_dense(A)
    L, D, W = numpy.linalg.svd(A_dense, full_matrices=False)
    return L[:, :k], D[:k], W[:k, :]


def test_case(A, k, s1=0, s2=0, s3=0, s4=0):
    # nnz_svd approximation
    L_nnz, D_nnz, W_nnz = nnz_svd(A, k, s1, s3, s2, s4, 1)
    A_nnz_svd_approx = numpy.dot(L_nnz, numpy.dot(numpy.diag(D_nnz), W_nnz))
    #numpy.zeros((L_nnz.shape[0], W_nnz.shape[0]), order='F')
    #elem.DiagonalScale(elem.RIGHT, elem.NORMAL, D_nnz, L_nnz)
    #elem.Gemm(elem.NORMAL,elem.TRANSPOSE,1.,L_nnz,W_nnz,0.,A_nnz_svd_approx)

    # standard svd approximation (reference solution)
    A_dense                   = kdt_to_numpy_dense(A)
    L_dense, D_dense, W_dense = numpy.linalg.svd(A_dense, full_matrices=False)
    L_dense                   = L_dense[:, :k]
    D_dense                   = D_dense[:k]
    W_dense                   = W_dense[:k, :]
    A_dense_svd_approx        = numpy.dot(numpy.dot(L_dense, numpy.diag(D_dense)), W_dense)

    """
    Kens quality measurement:

       ( ||A||_F ^2 - || A - A^S_k ||_F^2 ) / || [A]_k ||_F^2

    - [A]_k is the best rank-k approximation, and
    - A^S_k is the sketching-based rank-k approximation.
    """
    error = (numpy.linalg.norm(A_dense, 'fro')**2 -
             numpy.linalg.norm(A_dense - A_nnz_svd_approx, 'fro')**2)
    error /= numpy.linalg.norm(A_dense_svd_approx, 'fro')**2

    return error

if __name__ == '__main__':
    # Note that the matrix needs to be big and the dimensionality reduction
    # needs to be big as well (see Figure 1 and 2).
    m  = 9000
    n  = 150
    s1 = int(m * 0.01)
    s2 = s1
    s3 = int(s1 * 0.1)
    s4 = s3
    matrix_rank = int(s4 * 0.3)
    A  = kdt_matrix(m, n, matrix_rank)

    comm = MPI.COMM_WORLD

    print 'Test params: m=%d, n=%d, rank=%d, s1=%d, s2=%d, s3=%d, s4=%d' \
               % (m, n, matrix_rank, s1, s2, s3, s4)
    if comm.Get_size() == 1:
        for k in range(matrix_rank, matrix_rank / 2, -1):
            error = test_case(A, k, s1, s2, s3, s4)
            print 'k=%d, (||A||_F^2 - ||A - A^S_k||_F^2) / ||[A]_k||_F^2 = %f' \
                            % (k, error)
    else:
        if comm.Get_rank() == 0:
            print 'supported only for 1-process runs'
