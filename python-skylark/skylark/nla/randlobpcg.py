

import numpy as np
import numpy.linalg as linalg

from scipy.sparse.linalg import lobpcg
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator

from skylark import sketch




def as2d(v):
    '''
    Makes sure that the posibly 1D vector v gets another bracketing level to
    actually behave as a 2D object/matrix column
    '''
    if v.ndim == 2:
        return v
    else:
        aux = np.array(v, copy=False)
        aux.shape = (v.shape[0], 1)
        return aux

    
class CallableLinearOperator(LinearOperator):
    '''
    Extends standard LinearOperator by defining () operator
    '''
    def __call__(self, x):
        return self.matmat(x)


def symmetrizer(A):
    '''
    Returns a (callable) linear operator Op with Op(something) = At * A * something
    '''
    n = min(A.shape)
    def mult(x):
        y = np.dot(A.T, np.dot(A, x))
        return as2d(y)
    return CallableLinearOperator((n ,n), matvec=mult, matmat=mult, dtype=np.float64)
    
    
def upper_triangular_preconditioner_symmetrizer(R):
    '''
    Returns a (callable) linear operator Op where Op(something) should represent the application of
    inverse(Rt * R) on something.
    '''
    m, n = R.shape
    assert (m == n)
    def precond(y):
        '''
        Find x in inverse(Rt*R) * y = x => Rt * (R * x) = y where R * x = z by:
        - Rt * z = y => z = ...
        - R  * x = z => x = ...
        i.e. two back-substitutions
        '''
        z = solve_triangular(R.T, y, lower=True)
        x = solve_triangular(R,   z, lower=False)
        return as2d(x)
    return CallableLinearOperator((m ,n), matvec=precond, matmat=precond, dtype=np.float64)



def lobpcg_randEVD(A, k, sketching_type, s):
    '''
    Approximate top k eigenvalue and corresponding right eigenvector computation 
    for At * A using LOBPCG over preconditioner attained through sketching.
    
    Parameters
    ----------
    A              : m x n matrix, expected tall-and-thin (m > n)
    k              : number of top eignevalues
    sketching_type : sketching transformation type (row number reducer)
    s              : number of rows in the sketched matrix
    
    Returns
    --------
    lambdas : top k eigenavalues of At * A
    Vt      : top k right singular vectors At * A (as rows of Vt)
    
    Notes/TODO
    -----------
    - ordering eigenvalues in lobpcg
    - opt out the option of using a generic symmetric eigenvalue solver internally
    
    '''
    m, n = A.shape
    assert m > n
    assert n >= k
    
    if s == None:
        s = 4 * n
    assert s < m
        
    sketch = sketching_type(m, s)
    B = sketch * A
    U, Sigma, Vt = linalg.svd(B, full_matrices=False)
    
    Q, R = linalg.qr(B) # alternatively: connection of R to V, Sigma

    Aop = symmetrizer(A)
    X = Vt[:k, :].T
    Rop = upper_triangular_preconditioner_symmetrizer(R)
    lambdas, Vt = lobpcg(Aop, X, M=Rop, largest=True)
    return lambdas, Vt



def power_iterations_randEVD(A, k, sketching_type, power_iters):
    '''
    Approximate top k eigenvalue and corresponding right eigenvector computation 
    for At * A using power iterations in a sketched SVD fashion.
    
    Parameters
    ----------
    A              : m x n matrix, expected tall-and-thin (m > n)
    k              : number of top eignevalues
    sketching_type : sketching transformation type (row number reducer)
    power_iters    : number power iteration steps towards approximate range extraction of A.
    
    Returns
    --------
    lambdas : top k eigenavalues of At * A
    Vt      : top k right singular vectors At * A (as rows of Vt)   
    '''    
    
    m, n = A.shape
    assert m > n
    assert n >= k
    sketch = sketching_type(n, k)
    
    # Y = A * Omega - or generally right sketch (Algorithm 4.3)
    Y = sketch / A
    
    # Apply (A * At) for power_iters times on Y from the left (Algorithm 4.3)
    for i in range(power_iters):
        Y = np.dot(A.T, Y)
        Y = np.dot(A, Y)
        
    # Q should now be a range approximation of A
    Q, R = linalg.qr(Y)
    
    # Algorithm 5.1 steps (SVD)
    B = np.dot(Q.T, A)
    U, Sigma, Vt = linalg.svd(B, full_matrices=False)
    
    # Singular value approximations for A as eigenvalue approximations for At * A
    lambdas = Sigma ** 2
    return lambdas, Vt



def generic_EVD(A, k):
    '''
    Top k eigenvalue and corresponding right eigenvector computation 
    for At * A using a generic approach.
    
    Parameters
    ----------
    A              : m x n matrix, expected tall-and-thin (m > n)
    k              : number of top eignevalues
    
    Returns
    --------
    lambdas : top k eigenavalues of At * A
    Vt      : top k right singular vectors At * A (as rows of Vt)
    '''
    AtA = np.dot(A.T, A)
    w, V = linalg.eig(AtA)
    indices = np.argsort(w)[::-1][:k]
    lambdas = w[indices]
    Vt = V[:, indices].T
    return lambdas, Vt


if __name__ == '__main__':
    # A is m x n 
    m = 30
    n = 5

    # Number of largest eigenvalues of At * A to approximate/compute
    k = 3

    # Sketching transform type
    sketching_type = sketch.JLT # or better sketch.FJLT

    # Target sketching size for the right sketching in  Halko, Martinsson, Tropp paper 
    s = 4 * n

    # Number of power iters in Algorithm 4.3 in the paper
    power_iters = 2

    # An example/test matrix
    A = np.random.random((m, n))


    # Computing top eigenvalues and corresponing right eigenvectors for the three cases
    w0, Vt0 = generic_EVD(A, k)
    w1, Vt1 = lobpcg_randEVD(A, k, sketching_type, s)
    w2, Vt2 = power_iterations_randEVD(A, k, sketching_type, power_iters)

