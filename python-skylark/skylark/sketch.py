import elem
import cskylark
from skylark import DISTMAT, DISTMAT_STR, SKETCHMAT_STR
import sprand
import numpy.random
import scipy.sparse 
import scipy.fftpack
import scipy.stats
from math import log, ceil
from scipy import sqrt
import matplotlib.pyplot as plt

class SkylarkError(Exception):
    pass

class SketchingTransform(object):
    def __init__(self, n, k, context=123):
        """
        A sketching transform in general is a map from R^n to R^k which preserves some properties.
        
        Parameters
        -----------
        n : input dimension
        k : sketching dimension
        context : either an integer seed or a Skylark context object
        
        Returns
        --------
        Sketch object
        """
        self.context = context
        self.dimensions = (k , n)
            
    def sketch(self, A, dimension = 1):
        """
        Sketch a matrix A
        
        Parameters
        -----------
        A : An numpy 2d array Or an Elemental Distributed matrix
        dimension : If dimension = 1, S is applied to columns of A, i.e. number of rows of S is sketched from m to s.  
                    If dimension = 2, S is applied to rows of S, i.e., number of rows of S is sketched from m to s.
         
        Returns
        ----------
        Sketched matrix.
        """  
        raise NotImplementedError
    
    
class JLT(SketchingTransform):
    """
    Class to represent classic Johnson-Lindenstrauss dense sketching using Gaussian Random maps. 

     Examples
        --------
        Let us bring *skylark* and other relevant Python packages into our environment.
        
        >>> import skylark, skylark.utilities, skylark.sketch
        >>> import scipy
        >>> import numpy.random
        >>> import matplotlib.pyplot as plt
        
        Let us generate some data, e.g., a data matrix whose entries are sampled uniformly from the interval [-1, +1].
        
        >>> n = 300
        >>> d = 1000
        >>> A = numpy.random.uniform(-1.0,1.0, (n,d))
        
        Create a sketch operator corresponding to Gaussian sketching.
        
        >>> seed = 123 
        >>> mysketch = skylark.sketch.JL(seed)
        
        Let us right-sketch A with Gaussian random matrix, to get 10% distortion with 95% probability. 
        
        >>> epsilon = 0.1
        >>> delta = 0.05
        >>> k = (epsilon, delta)
        >>> B = mysketch.sketch(A, k, 'right')
        
        Let us compute norms of the row-vectors before and after sketching.
        
        >>> norms_A = skylark.utilities.norms(A)
        >>> norms_B = skylark.utilities.norms(B)
         
        
        Plot the histogram of distortions (ratio of norms for original to sketched vectors).

        >>> distortions = scipy.ravel(norms_A/norms_B)
        >>> plt.hist(distortions,10)
        >>> plt.show()   
        
"""
    
    def __init__(self, n, k, context): 
        SketchingTransform.__init__(self, n, k, context)
        if isinstance(self.context, int):
            numpy.random.seed(self.context)
            self.SketchingOperator = numpy.matrix(numpy.random.randn(k, n)/sqrt(k))
        if isinstance(self.context, cskylark.Context): 
            self.SketchingOperator = cskylark.JLT(context, DISTMAT_STR , SKETCHMAT_STR, n, k)
        
    def sketch(self, A, dimension = 1):
        """
        sketch(A, k, dimension=0)

        Sketch an m-by-n matrix A to k = k(epsilon, delta) dimensions, 
        by row or column, using classic Gaussian random maps.

        The sketching transform is based on a variations of the classic Johnson-Lindenstrauss Lemma [1]_. 
        In particular, conceptually we use the construction of [2]_, which generalizes the results of 
        [3]_ and [4]_. Note that the sparse sketch of [4]_ is implemented in SparseJL class.


        Parameters
        ----------
        A : m-x-n matrix
            numpy ndarray or matrix
        k : tuple of floats, or int
            Dimension of the low-dimensional sketched data - can either be an integer 
            or a tuple (epsilon, delta) implicitly specifying :math:`k = 2*log(2/\delta)/\epsilon^2`
        dimension : 'left' | 'right' 
            left or right sketch
        

        Returns
        -----------
        
        For left sketch, returns a k x n sketched matrix
        for right sketch, returns a m x k sketched matrix

        
        Notes
        -----
        Theorem 3.1 in [2]_: For any integer n, :math:`\epsilon \in (0,0.5], \delta \in (0,1)`, 
        set :math:`k=C \\epsilon^{-2} \\log(2\\delta^{-1})` where C is a constant. Define
        a random map :math:`T: R^n \mapsto R^k : T(x) = \\frac{1}{\\sqrt{k}} R x` where  
        R is a k-by-n random matrix whose elements are independent random variables with zero-mean
        and unit variance, and have uniform subgaussian tails, then for all :math:`x\in R^n`,  
        with probability atleast :math:`1-\\delta`, we have that 
        
        .. math:: (1-\epsilon) \|x\|_2 \leq \|Tx\|_2 \leq (1+\epsilon) \|x\|_2
        
        Gaussian or Bernoulli random variables satisfy subgaussianity.  Note that the above event also 
        guarantees that pairwise distances between points in a pointset are also preserved. 

         References
        ----------
        .. [1] W. B. Johnson and J. Lindenstrauss, Extensions of Lipschitz 
               mappings into a Hilbert Space, Contemp Math 26 (1984), 189-206
        .. [2] Jiri Matousek, On variants of the Johnson-Lindenstrauss Lemma, 
               Random Structures and Algorithms, Vol 33, Issue 2.
        .. [3] P. Indyk and R. Motwani, Approximate nearest neighbors: Towards removing the curse of dimensionality, 
               Proc 30th ACM Symp Theory of Computing, 1998, pp 604-613.
        .. [4] D. Achlioptas, Database-friendly random projections: 
               Johnson-Lindenstrauss with binary coins, JCSS 66 (2003), 671-687.
                              
        """
        SA = None
        
        if isinstance(A, numpy.ndarray):
            m,n = A.shape
            if dimension==1:
                if self.dimensions[1] != m:
                        raise SkylarkError("Incompatible dimensions")
                SA = self.SketchingOperator*A # we want to do this more efficiently than dgemm
                
            if dimension==2:   
                if self.dimensions[1] != n:
                        raise SkylarkError("Incompatible dimensions")
                SA = A*self.SketchingOperator.T
        
        if isinstance(A, DISTMAT):
            SA = elem.Mat()
            m = A.Height()
            n = A.Width() 
            k = self.dimensions[0]
            if dimension==1:
                if self.dimensions[1] != m:
                        raise SkylarkError("Incompatible dimensions")
                SA.Resize(k, n)
            if dimension==2:
                if self.dimensions[1] != n:
                        raise SkylarkError("Incompatible dimensions")
                SA.Resize(m, k)
            print "Applying..."
            print A, SA
            self.SketchingOperator.Apply(A, SA, dimension)
         
        return SA
    
    
class FJLT(object):
    """
    Class to represent classic Johnson-Lindenstrauss dense sketching using Gaussian Random maps. 

     Examples
        --------
        Let us bring *skylark* and other relevant Python packages into our environment.
        
        >>> import skylark, skylark.utilities, skylark.sketch
        >>> import scipy
        >>> import numpy.random
        >>> import matplotlib.pyplot as plt
        
        Let us generate some data, e.g., a data matrix whose entries are sampled uniformly from the interval [-1, +1].
        
        >>> n = 300
        >>> d = 1000
        >>> A = numpy.random.uniform(-1.0,1.0, (n,d))
        
        Create a sketch operator corresponding to Gaussian sketching.
        
        >>> seed = 123 
        >>> mysketch = skylark.sketch.FJLT(seed)
        
        Let us right-sketch A with Gaussian random matrix, to get 10% distortion with 95% probability. 
        
        >>> k = 100
        >>> B = mysketch.sketch(A, k, 'right')
        
        Let us compute norms of the row-vectors before and after sketching.
        
        >>> norms_A = skylark.utilities.norms(A)
        >>> norms_B = skylark.utilities.norms(B)
         
        
        Plot the histogram of distortions (ratio of norms for original to sketched vectors).

        >>> distortions = scipy.ravel(norms_A/norms_B)
        >>> plt.hist(distortions,10)
        >>> plt.show()   
        
"""
    
    def __init__(self, seed): 
        self.seed = seed 
    
    
    def sketch(self, A, k, dimension="left"):
        """
        Fast Johnson Lindenstrauss Transform (Ailon and Chazelle, 2009 [5]_)
        
        Currently uses hashing based sketch for the "P" matrix in P*H*D. 
        Such a sketch also works as shown by Matousek. 
 
        Parameters
        ----------
        A : m-x-n matrix
            numpy ndarray or matrix
        k : integer
            Dimension of the low-dimensional sketched data 
        dimension : 'left' | 'right' 
            left or right sketch
        

        Returns
        -----------
        For left sketch, returns a k x n sketched matrix
        for right sketch, returns a m x k sketched matrix
        
        References
        ----------
        .. [5] The Fast Johnson-Lindenstrauss Transform and Approximate Nearest Neighbors, 
            N. Ailon, B. Chazelle, SIAM J. Comput. 39 (2009), 302-322.
            
        To do
        ------------
        1. Do we need to allow iterating a few times?
        2. Need to implement sampling.
             
        """
        numpy.random.seed(self.seed)
        m,n = A.shape
        if dimension=="left":  
            d = scipy.stats.rv_discrete(values=([-1,1], [0.5,0.5]), name = 'uniform').rvs(size=m)
            D = scipy.sparse.spdiags(d, 0, m, m)
            B = D*A               
            B = scipy.fftpack.dct(B, axis =  0, norm = 'ortho')
        if dimension=="right":
            d = scipy.stats.rv_discrete(values=([-1,1], [0.5,0.5]), name = 'uniform').rvs(size=n)
            D = scipy.sparse.spdiags(d, 0, n, n)
            B = A*D
            B = scipy.fftpack.dct(B, axis = -1, norm = 'ortho')
        
        spsketch = SparseJL(self.seed*123, "hash")
        B = spsketch.sketch(B,k, dimension)
        
        return B
        
        
class SparseJL(object):
    """ 
    Class to represent Sparse Sketching matrices.
    
    Two types are provided:
 
        #. **Sign** : based on sparse +1/-1 valued matrices.
        #. **Hash** : based on countmin sketch as used in Clarkson/Woodruff. 

     Examples
        --------
        Let us bring *skylark* and other relevant Python packages into our environment.
        
        >>> import skylark, skylark.utilities, skylark.sketch
        >>> import scipy
        >>> import numpy.random
        >>> import matplotlib.pyplot as plt
        
        Let us generate some data, e.g., a data matrix whose entries are sampled uniformly from the interval [-1, +1].
        
        >>> n = 300
        >>> d = 1000
        >>> A = numpy.random.uniform(-1.0,1.0, (n,d))
        
        Create a sketch operator corresponding to Gaussian sketching.
        
        >>> seed = 123 
        >>> mysketch = skylark.sketch.SparseJL(seed, "hash")
        
        Let us right-sketch A with Gaussian random matrix, to get 10% distortion with 95% probability. 
        
        >>> k = 100
        >>> B = mysketch.sketch(A, k, 'right')
        
        Let us compute norms of the row-vectors before and after sketching.
        
        >>> norms_A = skylark.utilities.norms(A)
        >>> norms_B = skylark.utilities.norms(B)
         
        
        Plot the histogram of distortions (ratio of norms for original to sketched vectors).

        >>> distortions = scipy.ravel(norms_A/norms_B)
        >>> plt.hist(distortions,10)
        >>> plt.show()    
    """

    def __init__(self, seed, type="hash"): 
        self.seed = seed
        self.type = type
        
    def sketch(self, A, k, dimension="left"):
        """
        Common sketch method that routes to either hash or sign based on type.
        """
        if self.type=="hash":
            B = self.hash(A, k, dimension=dimension)
        if self.type=="sign":
            B = self.sign(A, k, dimension=dimension)
        return B
    
        
    def sign(self, A, k, q=1.0/3.0, dimension="left"):
        """
        sign(A, k, dimension=0)

        Sketch an m-by-n matrix A to k = k(epsilon, delta) dimensions, 
        by row or column, using sign random maps.

        The sketching transform is based on the construction. 
        
        The elements of the sketching matrix S are independently sampled from 
        {0, +1, -1} with probabilities 2/3, 1/6 and 1/6 respectively. 
         
        Parameters
        ----------
        A : m-x-n matrix
            numpy ndarray or matrix
        k : tuple of floats, or int
            Dimension of the low    -dimensional sketched data - can either be an integer 
            or a tuple (epsilon, delta) implicitly specifying :math:`k = 2*log(2/\delta)/\epsilon^2`
        dimension : "left" | "right" 
            left or right sketch
        

        Returns
        -----------
        
        For left sketch, returns a k x n sketched matrix
        for right sketch, returns a m x k sketched matrix
       """
         
        numpy.random.seed(self.seed)
        m,n = A.shape
        if isinstance(k, tuple):
            (epsilon, delta) = k
            #k = (4 + 2*delta)/(epsilon*epsilon*(1.0/2.0 - epsilon/3.0))
            k = 2/(epsilon*epsilon)
            if dimension=="left":
                k = ceil(k*log(n))
            else:
                k = ceil(k*log(m))
            
        # Achlioptas/Matsouek construction 
        density = q 
        nz_values = [-sqrt(1.0/q), +sqrt(1.0/q)]
        nz_prob_dist = [0.5, 0.5]
        
        if dimension=="left":    
            S = sprand.sample(k, m, density, nz_values, nz_prob_dist)/sqrt(k)
            B = S*A # we want to do this more efficiently than dgemm
        
        if dimension=="right":
            S = sprand.sample(n, k, density, nz_values, nz_prob_dist)/sqrt(k) # we want to do this more efficiently than dgemm
            B = A*S
         
        B = B*sqrt(1.0/q) # we seem to need this according to experiments, but not theory!! please check. 
        return B
            
        
    def hash(self, A, k, q = 1, dimension="left"):
        """
        Sketch an m-by-n matrix A to k dimensions, 
        by row or column, using hashing based sketching.

        The sketching transform is based on the construction given in Clarkson/Woodruff. 
        
        The non-zero elements of the sketching matrix have value q with equal probability. 
         
        Parameters
        ----------
        A : m-x-n matrix
            numpy ndarray or matrix
        k : int
            Dimension of the low-dimensional sketched data
        q : int
            Hash value (non-zero entries in the sketch are +q, -q with equal probability)
        dimension : "left" | "right" 
            left or right sketch
            
        Returns
        -----------
        For left sketch, returns a k x n sketched matrix
        for right sketch, returns a m x k sketched matrix
        
        """
        numpy.random.seed(self.seed)
        m,n = A.shape
        
        if dimension == "left":
            S = sprand.hashmap(k, m, dimension=0, nz_values=[-q,+q], nz_prob_dist=[0.5,0.5])
            B = S*A
        if dimension == "right":
            S = sprand.hashmap(k, n, dimension=1, nz_values=[-q,+q], nz_prob_dist=[0.5,0.5])
            B = A*S
        return B
    
    
class Sampling(object):
    """
    Class to represent sampling-based sketching. (incomplete). 
    """
    def __init__(self, seed, probdist):
        self.seed = seed
        self.probdist = probdist

    def sketch(self, A, k, dimension="left"):
        """
        
        """
     

if __name__== "__main__":
        # generate a matrix 
        n = 100
        d = 1000
        
        A = numpy.random.uniform(-1.0,1.0, (n,d))
    
        #Let us sketch this matrix with 10% distortion with 95% confidence.
        seed = 123 
        
        epsilon = 0.1
        delta = 0.05
        #k = (epsilon, delta)
        k=200
        
        sketcher = FJLT(seed)
        B = sketcher.sketch(A, k, dimension = "right")
        #B = sketcher.sign(A, k, 1)
        
        print "size of sketched matrix ", B.shape
        
        
        #Let us check if norms are preserved. 
        norms_A = numpy.dot((A*A), numpy.ones((A.shape[1],1)))
        norms_B = numpy.dot((B*B), numpy.ones((B.shape[1],1)))
        
        #D = scipy.sparse.diags(scipy.ones(n), 0) 
        #distances_A = euclidean(A,A) + D
        #distances_B = euclidean(B,B) + D
        distortions = scipy.ravel(norms_A/norms_B)    
        plt.hist(distortions, 50)
        plt.show()
    

