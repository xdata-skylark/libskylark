import unittest
import numpy as np
from scipy.sparse import rand as rand_matrix
from scipy import linalg
import El

import skylark.nla as slnla
from ctypes import cdll, c_bool
from .. import utils

skylarklib = cdll.LoadLibrary('libcskylark.so')
skylarklib.sl_has_elemental.restype = c_bool
skylarklib.sl_has_combblas.restype  = c_bool


class NLATestCase(unittest.TestCase):
    """Tests nla functions."""
    def test_approximate_svd(self):
        """Compute the SVD of **A** such that **SVD(A) = U S V^T**."""

        # Generate random matrix
        A = rand_matrix(200, 200, density=0.01, format='csc', dtype=np.float64, \
                        random_state=None)

        # Dimension to apply along.
        k = 50

        U = El.Matrix()
        S = El.Matrix()
        V = El.Matrix()

        slnla.approximate_svd(A, U, S, V, k = k)
        S = S.ToNumPy()

        # Calculate using scipy        
        Us, Ss, Vs = linalg.svd(A.todense())

        # Checking ${precison} decimals
        precision = 4
        for i in xrange(5):
            # S is a Column matrix, Ss a vector
            diff = round(S[i][0] - Ss[i], precision)
            self.assertEqual(diff, 0)


    
    def test_approximate_symmetric_svd(self):
        """Compute the SVD of symmetric **A** such that **SVD(A) = V S V^T**"""
        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(NLATestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)