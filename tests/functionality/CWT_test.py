import unittest
import numpy as np
from mpi4py import MPI

from helper.elemental_matrix import create_elemental_matrix
from skylark import cskylark

import elem

class CWT_test(unittest.TestCase):

    def setUp(self):
        # Create a matrix to sketch
        elem.Initialize()
        self.A = create_elemental_matrix(10, 5)

        # Initilize context
        self.ctxt = cskylark.Context(123834)

        #FIXME:
        self.exp_col_norm = 24.7991935353
        self.exp_row_norm = 26.3628526529

    def tearDown(self):
        self.A.Free()
        self.ctxt.Free()
        elem.Finalize()

    #FIXME: how many applications/averages
    def test_apply_colwise(self):
        S  = cskylark.CWT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", 10, 6)
        SA = elem.Mat()
        SA.Resize(6, 5)
        S.Apply(self.A, SA, 1)

        self.assertAlmostEqual(np.linalg.norm(SA.Data()), self.exp_col_norm, places=5)

    #FIXME: how many applications/averages
    def test_apply_rowwise(self):
        S  = cskylark.CWT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", 5, 3)
        SA = elem.Mat()
        SA.Resize(10, 3)
        S.Apply(self.A, SA, 2)

        self.assertAlmostEqual(np.linalg.norm(SA.Data()), self.exp_row_norm, places=5)


if __name__ == '__main__':
    unittest.main()
