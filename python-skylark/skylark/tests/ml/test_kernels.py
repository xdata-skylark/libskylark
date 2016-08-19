import unittest
import numpy as np

import skylark.io as sl_io
import skylark.ml.kernels as sl_kernels
import El

from skylark.tests.utils import *

# We will test our kernels using the data from a testfile.
# Make sure that the file is accessible! 
fname = "../../datasets/usps.t"


class LinearTestCase(unittest.TestCase):
    """Tests linear kernel over d dimensional vectors."""

    def test_Linear_gram(self):
        """Tests the dense Gram matrix evaluated over the datapoints."""
        pass
    
    def test_Linear_rft(self):
        """Create random features transform for the kernel."""
        pass


class GaussianTestCase(unittest.TestCase):
    """Tests Gaussian kernel over d dimensional vectors."""

    def test_Gaussian_gram(self):
        """Tests the dense Gram matrix evaluated over the datapoints."""
        
        # Read the data
        X = El.DistMatrix()
        L = El.DistMatrix()
        sl_io.readlibsvm('data/usps.t', X, L, 0)

        
        # Execute the kernel
        K = El.DistMatrix()
        sigma = np.random.randint(9) + 1 # Sigma between 1 and 10
        Gaussian_kernel = sl_kernels.Gaussian(X.Height(), sigma)
        Gaussian_kernel.gram(X, K, 0, 0)

        # Check that K is a square matrix with dim  X.Height and X.Width
        self.assertTrue(K.Height() == K.Width())
        self.assertTrue(X.Height() == K.Height())
        self.assertTrue(X.Width() == K.Width())

        # Check that the diagonal contains 1
        for i in xrange(K.Height()):
            self.assertTrue(K.Get(i,i) == 1)

    def test_Gaussian_rft(self):
        """Create random features transform for the kernel."""
        # TODO: How should we check that?
        pass


class LaplacianTestCase(unittest.TestCase):
    """Tests Laplacian kernel over d dimensional vectors."""

    def test_Laplacian_gram(self):
        """Tests the dense Gram matrix evaluated over the datapoints."""
        
        # Read the data
        X = El.DistMatrix()
        L = El.DistMatrix()
        sl_io.readlibsvm('data/usps.t', X, L, 0)

        
        # Execute the kernel
        K = El.DistMatrix()
        sigma = np.random.randint(9) + 1 # Sigma between 1 and 10
        Laplacian_kernel = sl_kernels.Laplacian(X.Height(), sigma)
        Laplacian_kernel.gram(X, K, 0, 0)

        # Check that K is a square matrix with dim  X.Height and X.Width
        self.assertTrue(K.Height() == K.Width())
        self.assertTrue(X.Height() == K.Height())
        self.assertTrue(X.Width() == K.Width())

        # Check that the diagonal contains 1
        for i in xrange(K.Height()):
            self.assertTrue(K.Get(i,i) == 1)


    def test_Laplacian_rft(self):
        """Create random features transform for the kernel."""
        # TODO: How should we check that?
        pass
      
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DistancesTestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)
