import unittest
import numpy as np

import skylark.ml.kernels as sl_kernels
import skylark.tests.utils as sl_test_utils
import El



# We will test our kernels using the data from a testfile.
# Make sure that the file is accessible! 
fpath = "usps.t"

class KernelTestCase(unittest.TestCase):
    """ Generic tests for kernels over d dimensional vectors.
        For each kernel we load the usps.t file and run the gram function.
    """

    def test_Linear_kernel(self):
        """Test Linear kernel."""
        X, Y = sl_test_utils.load_libsvm_file(fpath, 0)
        K = El.DistMatrix()
        Linear_kernel = sl_kernels.Linear(X.Height())
        Linear_kernel.gram(X, K, 0, 0)

        self.assertTrue(sl_test_utils.is_kernel(K))
    
    def test_Gaussian_kernel(self):
        """Test Gaussian kernel."""
        X, Y = sl_test_utils.load_libsvm_file(fpath, 0)
        K = El.DistMatrix()
        Gaussian_kernel = sl_kernels.Gaussian(X.Height(), 10.0)
        Gaussian_kernel.gram(X, K, 0, 0)

        self.assertTrue(sl_test_utils.is_kernel(K))

    def test_Laplacian_kernel(self):
        """Test Laplacian kernel."""
        X, Y = sl_test_utils.load_libsvm_file(fpath, 0)
        K = El.DistMatrix()
        Laplacian_kernel = sl_kernels.Laplacian(X.Height(), 2.0)
        Laplacian_kernel.gram(X, K, 0, 0)

        self.assertTrue(sl_test_utils.is_kernel(K))
    
    def test_Polynomial_kernel(self):
        """Test Polynomial kernel."""
        X, Y = sl_test_utils.load_libsvm_file(fpath, 0)
        K = El.DistMatrix()
        Polynomial_kernel = sl_kernels.Polynomial(X.Height(), 1, 1, 1)
        Polynomial_kernel.gram(X, K, 0, 0)

        self.assertTrue(sl_test_utils.is_kernel(K, positive=False))
        
    def test_ExpSemiGroup_kernel(self):
        """Test ExpSemiGroup kernel."""
        X, Y = sl_test_utils.load_libsvm_file(fpath, 0)
        K = El.DistMatrix()
        ExpSemiGroup_kernel = sl_kernels.ExpSemiGroup(X.Height(), 0.1)
        ExpSemiGroup_kernel.gram(X, K, 0, 0)

        self.assertTrue(sl_test_utils.is_kernel(K))
    
    # TODO: Test the Matern Kenrel (after implement it in C++)
    
    # TODO: Test all RFT

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DistancesTestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)
