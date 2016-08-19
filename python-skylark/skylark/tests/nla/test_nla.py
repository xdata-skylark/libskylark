import unittest
import numpy as np
import El

import skylark.nla as sl_nla
from .. import utils



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
        m = 500
        n = 100

        # Generate problem
        A, B, X, X_opt= (El.Matrix(), El.Matrix(), El.Matrix(), El.Matrix())
        El.Gaussian(A, m, n)
        El.Gaussian(X_opt, n, 1)
        El.Zeros(B, m, 1)
        El.Gemm( El.NORMAL, El.NORMAL, 1, A, X_opt, 0, B);

        # Solve it using faster least squares
        sl_nla.faster_least_squares(A, B, X)

        # Check the norm of our solution
        El.Gemm( El.NORMAL, El.NORMAL, 1, A, X, -1, B);
        self.assertAlmostEqual(El.Norm(B), 0)

        # Checking the solution
        self.assertTrue(utils.equal(X_opt, X))
      
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NLATestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)