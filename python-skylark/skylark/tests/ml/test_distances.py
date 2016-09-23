import unittest
import numpy as np
from scipy.sparse import rand as rand_matrix

import skylark.ml.distances as sl_ml_distances
from .. import utils




class DistancesTestCase(unittest.TestCase):
    """Tests distances functions."""

    def test__multiply_numpy(self):
        """
        Element-wise mult of X and Y that works with X and Y numpy matrices
        """
        A = np.random.rand(10,10)
        B = np.random.rand(10,10)
        C = sl_ml_distances._multiply(A, B)
        Cnumpy = np.multiply(A, B)

        self.assertTrue(np.array_equal(C, Cnumpy))


    def test__multiply_scipy(self):
        """
        Element-wise mult of X and Y that works with X and Y scipy matrices
        """

        # Check for csc
        A = rand_matrix(200, 200, density=0.01, format='csc')
        B = rand_matrix(200, 200, density=0.01, format='csc')

        C_csc = sl_ml_distances._multiply(A, B)
        C_csc_scipy = A.multiply(B)
        
        self.assertTrue((C_csc!=C_csc_scipy).nnz==0)

        # Check for csr
        A = rand_matrix(200, 200, density=0.01, format='csr')
        B = rand_matrix(200, 200, density=0.01, format='csr')

        C_csr = sl_ml_distances._multiply(A, B)
        C_csr_scipy = A.multiply(B)
        
        self.assertTrue((C_csr!=C_csr_scipy).nnz==0)


    def test_euclidean(self):
        """Create a euclidean distance matrix (actually returns distance squared)"""
        A = np.array([[1,2],[3,4]])
        B = np.array([[5,6],[7,8]])
        C = sl_ml_distances.euclidean(A,B)
        C_res = np.array([[32, 8],[72, 32]])
        self.assertTrue(np.array_equal(C, C_res))
       
      
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DistancesTestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)
