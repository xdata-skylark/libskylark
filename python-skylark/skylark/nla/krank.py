'''
Implementation of all algorithms for constructing approximate matrix
decompositions as described in [1]_ An approximate matrix decomposition is
generally constructed in two stages: 

 * In the first stage, an orthonormal matrix Q is constructed by a randomization  
   technique that aims at identifying a subspace capturing most of the action of
   the input matrix A; this stage is typically implemented by two substages,
   namely the sketching of A followed by a QR decomposition of the sketched
   matrix. This stage reflects in our ``RandomizedRangeFinder`` class. 
 
 * In the second stage the orthonormal matrix is used for projecting A and
   subsequently decomposing the projection with a standard (non-randomized)
   method. We provide for singular value decomposition (SVD) and eigenvalue
   decomposition (EVD) in ``RangeAssistedSVD`` and ``RangeAssistedEVD`` classes
   respectively.   

.. [1] Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp, "Finding structure
       with randomness: Probabilistic algorithms for constructing approximate
       matrix decompositions", http://arxiv.org/abs/0909.4061 
'''

import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import warnings
import scipy.linalg as slinalg

# FIXME: raise ImportError exception (possibly better at the point of use) in
# case scipy.linalg.interpolative module is not available.
try: 
    import scipy.linalg.interpolative as sli
except ImportError:
    print 'Row-extraction methods are not supported'

# FIXME: consider scaling of sketching matrices


def SRFT_matrix(n, s):
    '''
    SRFT (Subsampled Random Fourier Transform) matrix. 

    This is a structured random matrix 
    :math:`\\Omega = \\sqrt{\\frac{n}{s}} D F R`.  
    
    Parameters
    ----------
    n : int
     Width of the input matrix (number of columns).

    s : int
     Width of the sketched matrix.
    
    Returns
    -------
    omega : numpy array
     :math:`\\Omega` matrix.    
    '''
    D = D_matrix(n)
    F = F_matrix(n)
    R = R_matrix(n, s)
    factor = np.sqrt((1.0 * n) / s)
    # FIXME: could do D * ... by row scalings  
    SRFT = factor * np.dot(D, np.dot(F, R))
    return SRFT


def D_matrix(n):
    '''
    Diagonal matrix with its diagonal entries independent random variables
    uniformly distributed on the complex unit circle. 

    Parameters
    ----------
    n : int
     Size of the diagonal matrix (n x n).

    Returns
    -------
    D : numpy array
     Diagonal matrix.
    '''
    thetas = random.uniform(0, 2 * np.pi, n)
    values = [np.exp(1.0j * theta) for theta in thetas]
    D = np.diag(values)
    return D        


def F_matrix(n):
    '''
    Unitary, discrete Fourier (DFT) matrix.
    
    Parameters
    ----------
    n : int
     Size of the unitary DFT matrix (n x n).
    
    Returns
    -------
    F : numpy array
     Unitary DFT matrix.
    '''
    p, q = np.mgrid[0:n, 0:n]
    F = np.sqrt(n) * np.exp((-2.0j * np.pi * p * q) / n)
    return F


def R_matrix(n, s):
    '''
    Matrix composed of randomly selected column of the identity matrix without
    replacement. 

    Parameters
    ----------
    n : int
     Size of the identity matrix (n x n).

    s : int
     Number of columns randomly selected.
    
    Returns
    -------
    R : numpy array
     Matrix of selected columns.
    '''
    data = range(n)
    samples = Fisher_Yates_samples(range(n), s)
    identity_matrix = np.eye(n)
    R = identity_matrix[:, samples]
    return R


def Fisher_Yates_samples(data, size):
    '''
    Sampling without replacement using the Fisher-Yates strategy.
    
    Parameters
    ----------
    data : numpy array
     Container to sample from.

    size : int
     Number of samples.

    Returns
    -------
    samples : numpy array
     Samples.
    '''
    n = len(data)
    assert n >= size
    samples = []
    k = n
    for i in range(size):
        index = random.randint(0, k)
        chosen = data[index]
        data[index] = data[k-1]
        data[k-1] = chosen
        samples.append(chosen)
        k-=1
    return samples


class RandomizedRangeFinder(object):
    '''
    Construct an orthonormal matrix of reduced size intended to capture most of
    the action of the input matrix, i.e. approximate its range, using
    randomization techniques.

    Attributes
    ----------
    args : dict
     Mapping of the randomization methods to their parameters.     
    '''
    args = {'generic'            : {'s':None},
            'adaptive'           : {'epsilon':None, 'r':None, 'max_iters':100},
            'power_iteration'    : {'s':None, 'q':1},
            'subspace_iteration' : {'s':None, 'q':1},
            'fast_generic'       : {'s':None}}
    
    def __init__(self, A, method, params, parallel=False):
        '''
        Class constructor.
        
        Parameters
        ----------
        A : numpy array
         Input matrix.

        method : string
          One of the following (see the keys in `args`):
           
           * ``'generic'``            : Algorithm 4.1 in [1]_
           * ``'adaptive'``           : Algorithm 4.2 in [1]
           * ``'power_iteration'``    : Algorithm 4.3 in [1]_
           * ``'subspace_iteration'`` : Algorithm 4.4 in [1]_
           * ``'fast_generic'``       : Algorithm 4.5 in [1]_

        params : dict (see the values in ``args``):
         Method depending parameters as a map with keys the parameter names
         (strings) and values the parameter values:
         
          * ``'generic'`` and ``'fast_generic'`` need the number of columns
            ``s`` of the orthonormal matrix (sketch size). 
          * ``'subspace_iteration'`` and ``'power_iteration'`` need the number
            of columns ``s`` of the orthonormal matrix (sketch size) and the
            number of iterations.
          * ``'adaptive'`` needs the tolerance ``epsilon``, the number of random 
            vectors ``r`` and maximum number of iterations permitted
            ``max_iters``.   
          
        parallel: {False, True}
         Boolean flag whether the computation of Q will run in parallel or
         not.
        
        Returns
        -------
        finder : object
         Ready to use object.

        Notes
        -----
        Orthonormal matrix Q computed will satisfy:
        :math:`||(I - Q Q^{*}) A)|| \\leq \\epsilon`
        with probability at least 
        :math:`1 - \\min\{m, r\} 10^{-r}`
        for an input matrix A of dimensions m x n.
        '''
        kwargs = RandomizedRangeFinder.args[method]
        for key, value in params.iteritems():
            kwargs[key] = value
        if None in kwargs.values():
            raise ValueError('Missing arguments')
        self.A = A
        self.method = method
        self.params = params
        self.parallel = parallel

        self.kwargs = kwargs
        self.__compute = getattr(self, '_RandomizedRangeFinder__%s' % method)

    
    def compute(self):
        '''
        Compute the orthonormal matrix Q.
        
        Parameters
        ----------
        None

        Returns
        -------
        Q : numpy array
         Orthonormal matrix Q approximating the range of input matrix A.
        '''
        return self.__compute()


    def __generic(self):
        A = self.A
        s = self.kwargs['s']

        m, n = A.shape
        S = random.randn(n, s)
        Y = np.dot(A, S)
        Q, R = linalg.qr(Y)
        return Q


    def __adaptive(self):
        A = self.A
        epsilon = self.kwargs['epsilon']
        r = self.kwargs['r']
        max_iters = self.kwargs['max_iters']

        m, n = A.shape
        w_list = [random.randn(n) for i in range(r)]
        y_list = [np.dot(A, w) for w in w_list]
        threshold = epsilon / (10. * np.sqrt(2. / np.pi))
        iters = 0
        j = 0
        Q = np.empty((m, 0))
        while max([linalg.norm(y) for y in y_list]) > threshold and iters < max_iters:
            j+=1
            y = y_list[j]
            np.dot(Q.T, y)
            y = y - np.dot(Q, np.dot(Q.T, y))
            q = y / linalg.norm(y)
            Q = np.hstack([Q, np.reshape(q, (m,1))])
            w = random.randn(n)
            z = np.dot(A, w)
            y = z - np.dot(Q, np.dot(Q.T, z))
            w_list.append(w)
            y_list.append(y)
            for i in range(j+1, j+r):
                y_list[i] = y_list[i] - np.dot(q, y_list[i]) * q
            iters +=1
        if iters == max_iters:
            warnings.warn('Failed to converge after %d iterations' % iters)
        return Q


    def __power_iteration(self):
         A = self.A
         s = self.kwargs['s']
         q = self.kwargs['q']

         m, n = A.shape
         S = random.randn(n, s)
         Y = np.dot(A, S)
         for i in range(q):
             Y = np.dot(A, np.dot(A.T, Y))
         Q, R = linalg.qr(Y)
         return Q


    def __subspace_iteration(self):
         A = self.A
         s = self.kwargs['s']
         q = self.kwargs['q']
         
         m, n = A.shape
         S = random.randn(n, s)
         Y = np.dot(A, S)
         Q, R = linalg.qr(Y)
         for i in range(q):
             Y = np.dot(A.T, Q)
             Q, R = linalg.qr(Y)
             Y = np.dot(A, Q)
             Q, R = linalg.qr(Y)
         return Q
    
 
    # TODO: see section 3.3 in "A fast randomized algorithm for the approximation
    # of matrices" for a fast implementation.
    def __fast_generic(self):
        A = self.A
        s = self.kwargs['s']
        
        m, n = A.shape
        SRFT = SRFT_matrix(n, s)
        Y = np.dot(A, SRFT)
        Q, R = linalg.qr(Y)
        return Q


class RangeAssistedSVD(object):
    '''
    Construct approximate SVD of a matrix as assisted by an approximation of its
    range. 
    
    Attributes
    ----------
    args : dict
     Mapping of the SVD construction methods to their parameters.     
    '''    
    args = {'direct'            : {},
            'row_extraction'    : {}}
    
    def __init__(self, A, Q, method, params, parallel=False):
        '''
        Class constructor.
        
        Parameters
        ----------
        A : numpy array
         Input matrix.
        
        Q : numpy array
         Orthonormal matrix approximating the range of the input matrix.
        
        method : string
         One of the following (see the keys in ``args``):
           
           * ``'direct'``             : Algorithm 5.1 in [1]_
           * ``'row_extraction'``     : Algorithm 5.2 in [1]_
        
        params : dict (see the values in ``args``):
         Method depending parameters as a map with keys the parameter names
         (strings) and values the parameter values; it is empty but retained for 
         reasons of uniformity with the interface of approximate eigenvalue
         decomposition.  
        
        parallel: {False, True}
         Boolean flag whether the computation of the approximate decomposition
         will run in parallel or not.
        
        Returns
        -------
        svd : object
         Ready to use object.
        '''
        kwargs = RangeAssistedSVD.args[method]
        for key, value in params.iteritems():
            kwargs[key] = value
        if None in kwargs.values():
            raise ValueError('Missing arguments')
        self.A = A
        self.Q = Q
        self.method = method
        self.params = params
        self.parallel = parallel

        self.kwargs = kwargs
        self.__compute = getattr(self, '_RangeAssistedSVD__%s' % method)

    
    def compute(self):
        '''
        Compute approximate SVD.
        
        Parameters
        ----------
        None

        Returns
        -------
        U : numpy array
         Matrix containing computed left eigenvectors as its columns.
 
        sigma : numpy array
         Vector with computed singular values.

        Vt : numpy array
         Matrix containing computed right eigenvectors as its rows.
        '''
        return self.__compute()


    def __direct(self):
        A = self.A
        Q = self.Q

        B = np.dot(Q.T, A)
        U, sigma, Vt = linalg.svd(B)
        U = np.dot(Q, U)
        return U, sigma, Vt


    def __row_extraction(self):
        A = self.A
        Q = self.Q

        m, n = A.shape
        m, k = Q.shape
        idx, proj = sli.interp_decomp(Q, k, rand=False)
        Qj = sli.reconstruct_skel_matrix(Q, k, idx)
        X = sli.reconstruct_interp_matrix(idx, proj)
        Qj = Qj.T
        X = X.T
        Aj = A[idx, :]
        W, R = linalg.qr(Aj.T)
        Z = np.dot(X, R.T)
        U, sigma, Vt = linalg.svd(Z)
        V = np.dot(W, Vt.T)
        Vt = V.T
        return U, sigma, Vt



class RangeAssistedEVD(object):
    '''
    Construct approximate EVD of a matrix as assisted by an approximation of its
    range. 
    
    Attributes
    ----------
    args : dict
     Mapping of the EVD construction methods to their parameters.     
    '''    
    args = {'direct'            : {},
            'row_extraction'    : {},
            'Nystrom'           : {},
            'one_pass'          : {'s':None}}
    
    def __init__(self, A, Q, method, params, parallel=False):
        '''
        Class constructor.
        
        Parameters
        ----------
        A : numpy array
         Input matrix.

        Q : numpy array
         Orthonormal matrix approximating the range of the input matrix.

        method : string
          One of the following (see the keys in ``args``):

           * ``'direct'``             : Algorithm 5.3 in [1]_
           * ``'row_extraction'``     : Algorithm 5.4 in [1]_
           * ``'Nystrom'``            : Algorithm 5.5 in [1]_
           * ``'one_pass'``           : Algorithm 5.6 in [1]_

        params : dict (see the values in ``args``):
         Method depending parameters as a map with keys the parameter names
        (strings) and values the parameter values; ``'one_pass'`` method needs
         the number of columns ``s`` of the orthonormal matrix (sketch size).
        
        parallel: {False, True}
         Boolean flag whether the computation of the approximate decomposition
         will run in parallel or not.
        
        Returns
        -------
        evd : object
         Ready to use object.
        '''

        kwargs = RangeAssistedEVD.args[method]
        for key, value in params.iteritems():
            kwargs[key] = value
        if None in kwargs.values():
            raise ValueError('Missing arguments')
        self.A = A
        self.Q = Q
        self.method = method
        self.params = params
        self.parallel = parallel

        self.kwargs = kwargs
        self.__compute = getattr(self, '_RangeAssistedEVD__%s' % method)

    
    def compute(self):
        '''
        Compute approximate EVD.
        
        Parameters
        ----------
        None

        Returns
        -------
        w : numpy array
         Vector with computed eigenvalues.
        
        U : numpy array
         Matrix containing computed eigenvectors as its columns.
        '''

        return self.__compute()


    def __direct(self):
        A = self.A
        Q = self.Q

        B = np.dot(Q.T, np.dot(A, Q))
        w, V = linalg.eig(B)
        U = np.dot(Q, V)
        return w, U


    def __row_extraction(self):
        A = self.A
        Q = self.Q

        m, n = A.shape
        m, k = Q.shape
        idx, proj = sli.interp_decomp(Q, k, rand=False)
        Qj = sli.reconstruct_skel_matrix(Q, k, idx)
        X = sli.reconstruct_interp_matrix(idx, proj)
        Qj = Qj.T
        X = X.T
        V, R = linalg.qr(X)
        Ajj = A[np.ix_(idx, idx)]
        Z = np.dot(R, np.dot(Ajj, R))
        w, W = linalg.eig(Z)
        U = np.dot(V, W)
        return w, U


    def __Nystrom(self):
        A = self.A
        Q = self.Q

        B1 = np.dot(A, Q)
        B2 = np.dot(Q.T, B1)
        C, lower = slinalg.cho_factor(B2)
        Ft = slinalg.cho_solve((C.T, lower), B1.T)
        F = Ft.T
        U, sigma, Vt = linalg.svd(F)
        w = sigma ** 2
        return w, U


    def __one_pass(self):
        A = self.A
        Q = self.Q
        s = self.kwargs['s']

        m, n = A.shape
        S = random.randn(n, s)
        Y = np.dot(A, S)
        Y = np.dot(Q, np.dot(Q.T, Y))
        B_approx, residuals, rank, s = linalg.lstsq(np.dot(S.T, Q), np.dot(Y.T, Q))
        B_approx = B_approx.T
        w, V = linalg.eig(B_approx)
        U = np.dot(Q, V)
        return w, U    


def randomized_SVD(A, k, q=1, parallel=False):
    '''
    Approximate SVD using a randomization technique. 

    This uses the following mix of methods for the two stages:
     
     * ``'subspace_iteration'`` for the randomization stage (computing Q). 
     * ``'direct'`` for the decomposition stage (computing SVD as assisted by
       Q). 
     
    Parameters
    ----------
    A : numpy array
     Input matrix.
    
    k : int
     Target rank.
    
    q : int
     Number of subspace iterations.
    
    parallel : {False, True}
     Boolean flag whether the computation of the approximate decomposition will
     run in parallel or not. 

    Returns
    -------
    U : numpy array
     Matrix containing computed left eigenvectors as its columns. 
    
    sigma : numpy array
     Vector with computed singular values.
    
    Vt : numpy array
     Matrix containing computed right eigenvectors as its rows.
    '''
    s = 2 * k
    
    # Compute the range finder
    params = {'s':s, 'q':q}
    rrf = RandomizedRangeFinder(A, 'subspace_iteration', params, parallel) 
    Q = rrf.compute()
    
    # Compute the range assisted approximate matrix decomposition
    params = {}
    raSVD = RangeAssistedSVD(A, Q, 'direct', params, parallel)
    U, sigma, Vt = raSVD.compute()
    return U, sigma, Vt


# sanity tests

def _range_finder_failure_test(A, arg_pool):
    failures = []
    for method, kwargs in RandomizedRangeFinder.args.iteritems():
        args = {}
        for key in kwargs.keys():
            args[key] = arg_pool[key]
        obj = RandomizedRangeFinder(A, method, args)
        try:
            result = obj.compute()
        except:
            failures.append(method)
    return failures


def _decomposition_failure_test(A, Q, klass, arg_pool):
    failures = []
    for method, kwargs in klass.args.iteritems():
        args = {}
        for key in kwargs.keys():
            args[key] = arg_pool[key]
        obj = klass(A, Q, method, args)
        try:
            result = obj.compute()
        except:
            failures.append(method)
    return failures


if __name__ == '__main__':
    # construction and method parameters
    height = 6
    width = 5
    pool = {
        's' : 3,
        'q' : 2,
        'epsilon' : 0.1,
        'r' : 10,
        'max_iters' : 30
        }
    
    A = np.random.rand(height, width)
    A = np.dot(A, A.T)
    
    # accumulate the method names that failed in approximating the range of A
    finder_failures = _range_finder_failure_test(A, pool)
    
    # compute an example Q using subspace iterations
    params = {'s': pool['s'], 'q': pool['q']}
    rrf = RandomizedRangeFinder(A, 'subspace_iteration', params)
    Q = rrf.compute()
    
    # accumulate the method names that failed in computing approximate SVD and
    # EVD of A  assisted by the previously computed  Q
    svd_failures    = _decomposition_failure_test(A, Q, RangeAssistedSVD, pool)
    evd_failures    = _decomposition_failure_test(A, Q, RangeAssistedEVD, pool)
    
    print finder_failures
    print svd_failures
    print evd_failures
    
