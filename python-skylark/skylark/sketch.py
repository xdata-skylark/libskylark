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
    def __init__(self, n, s, context=123):
        """
        A sketching transform - in very general terms - is a dimensionality-reducing map 
        from R^n to R^s which preserves key structural properties.
        
        Parameters
        -----------
        n : input dimension
        s : sketching dimension
        context : either an integer seed (default 123) or a Skylark Context object
        
        Returns
        --------
        A sketch object with properties:  context and shape (s,n) attributes.
        """
        self.context = context
        self.shape = (s , n)
            
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
        Here we demonstrate a non-distributed usage implemented using numpy arrays.  
        In C-api section on accessing lower layers we give an example of distributed JLT transform operating on Elemental matrices.
        
        >>> import skylark, skylark.utilities, skylark.sketch
        >>> import scipy
        >>> import numpy.random
        >>> import matplotlib.pyplot as plt
        
        Let us generate some data, e.g., a data matrix whose entries are sampled uniformly from the interval [-1, +1].
        
        >>> n = 300
        >>> d = 1000
        >>> A = numpy.random.uniform(-1.0,1.0, (n,d))
        
        Create a sketch operator corresponding to JLT sketching from d=1000 to s=100.
        
        >>> seed = 123 
        >>> s = 100 
        >>> mysketch = skylark.sketch.JLT(d, s, seed)
        
        Let us sketch A row-wise:
        
        >>> B = mysketch.sketch(A, 2)
        
        Let us compute norms of the row-vectors before and after sketching.
        
        >>> norms_A = skylark.utilities.norms(A)
        >>> norms_B = skylark.utilities.norms(B)
         
        
        Plot the histogram of distortions (ratio of norms for original to sketched vectors).

        >>> distortions = scipy.ravel(norms_A/norms_B)
        >>> plt.hist(distortions,10)
        >>> plt.show()   
        
"""
    
    def __init__(self, n, s, context=123): 
        SketchingTransform.__init__(self, n, s, context)
        if isinstance(self.context, int):
            numpy.random.seed(self.context)
            self.SketchingOperator = numpy.matrix(numpy.random.randn(s, n)/sqrt(s))
        if isinstance(self.context, cskylark.Context): 
            self.SketchingOperator = cskylark.JLT(context, DISTMAT_STR , SKETCHMAT_STR, n, s)
        
    def sketch(self, A, dimension = 1):
        """
        sketch(A, dimension=1)

        Sketch an m-by-n matrix A to s dimensions using JLT transform.

        Parameters
        ----------
        A : m-x-n matrix
            numpy ndarray or matrix
        dimension: 0 (columnwise), 1 (rowwise)

        Returns
        -----------
        
        For dimension=0, returns a s x n sketched matrix
        for dimension=1, returns a m x s sketched matrix
                               
        """
        SA = None
        
        if not(isinstance(A, DISTMAT)):
            m,n = A.shape
            if dimension==1:
                if self.shape[1] != m:
                        raise SkylarkError("Incompatible dimensions")
                SA = self.SketchingOperator*A # we want to do this more efficiently than dgemm
                
            if dimension==2:   
                if self.shape[1] != n:
                        raise SkylarkError("Incompatible dimensions")
                SA = A*self.SketchingOperator.T
        
        if isinstance(A, DISTMAT):
            SA = elem.Mat()
            m = A.Height()
            n = A.Width() 
            k = self.shape[0]
            if dimension==1:
                if self.shape[1] != m:
                        raise SkylarkError("Incompatible dimensions")
                SA.Resize(k, n)
            if dimension==2:
                if self.shape[1] != n:
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
        Fast Johnson Lindenstrauss Transform
        
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
    
