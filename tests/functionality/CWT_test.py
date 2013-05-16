import unittest
import numpy as np

from helper.elemental_matrix import create_elemental_matrix
from skylark import cskylark

class CWT_test(unittest.TestCase):

    def setUp(self):
        # Create a matrix to sketch
        self.A = create_elemental_matrix(10, 5)

        # Initilize context
        self.ctxt = cskylark.Context(123834)

        self.exp_col_norm = 10.0
        self.exp_row_norm = 5.0


    def test_apply_colwise(self):
        self.S = cskylark.CWT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", 6, 5)
        self.assertIsInstance(self.S, int)
        SA = elem.Mat()
        SA.Resize(6, 5)
        self.S.Apply(self.A, SA, 1)

        print np.norm(SA)
        self.assertEqual(np.norm(SA), self.exp_col_norm)

    def test_apply_rowwise(self):
        self.S = cskylark.CWT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", 5, 3)
        self.assertIsInstance(self.S, int)
        SA = elem.Mat()
        SA.Resize(10, 3)
        self.S.Apply(self.A, SA, 2)

        print np.norm(SA)
        self.assertEqual(np.norm(SA), self.exp_row_norm)


if __name__ == '__main__':
    unittest.main()
