import unittest
import numpy as np
from scipy.sparse import rand as rand_matrix
from scipy import linalg
import El

import skylark.nla as sl_nla
from ctypes import cdll, c_bool
from .. import utils

skylarklib = cdll.LoadLibrary('libcskylark.so')
skylarklib.sl_has_elemental.restype = c_bool
skylarklib.sl_has_combblas.restype  = c_bool


class NLATestCase(unittest.TestCase):
    """Tests nla functions."""
    def test_approximate_svd(self):
        """Compute the SVD of **A** such that **SVD(A) = U S V^T**."""

        n = 100

        # Generate random matrix
        A = El.DistMatrix()
        El.Uniform(A, n, n)
        A = A.Matrix()

        # Dimension to apply along.
        k = n

        U = El.Matrix()
        S = El.Matrix()
        V = El.Matrix()

        sl_nla.approximate_svd(A, U, S, V, k = k)

        # Check result
        RESULT = El.Matrix()
        El.Zeros(RESULT, n, n)

        El.DiagonalScale( El.RIGHT, El.NORMAL, S, U );
        El.Gemm( El.NORMAL, El.ADJOINT, 1, U, V, 1, RESULT );

        self.assertTrue(utils.equal(A, RESULT))


    
    def test_approximate_symmetric_svd(self):
        """Compute the SVD of symmetric **A** such that **SVD(A) = V S V^T**"""
        n = 100
        A = El.DistMatrix()
        El.Uniform(A, n, n)
        A = A.Matrix()

        # Make A symmetric
        for i in xrange(0, A.Height()):
            for j in xrange(0, i+1):
                A.Set(j,i, A.Get(i,j))

        # Usign symmetric SVD
        SA = El.Matrix()
        VA = El.Matrix()
        
        sl_nla.approximate_symmetric_svd(A, SA, VA, k = n)

        # Check result
        VAT = El.Matrix()
        El.Copy(VA, VAT)

        RESULT = El.Matrix()
        El.Zeros(RESULT, n, n)

        El.DiagonalScale( El.RIGHT, El.NORMAL, SA, VAT );
        El.Gemm( El.NORMAL, El.ADJOINT, 1, VAT, VA, 1, RESULT );

        self.assertTrue(utils.equal(A, RESULT))


    def test_faster_least_squares_NORMAL(self):
        """Solution to argmin_X ||A * X - B||_F"""
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NLATestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)