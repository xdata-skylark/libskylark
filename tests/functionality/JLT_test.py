import unittest
import itertools
import random
import numpy as np

from mpi4py import MPI
from skylark import cskylark
from helper.elemental_matrix import create_elemental_matrix

import elem

class JLT_test(unittest.TestCase):

    def setUp(self):
        elem.Initialize()
        self.A = create_elemental_matrix(10, 5)

        #FIXME: expected norm = ?
        self.exp_col_norm = 26.97375583727611
        self.exp_row_norm = 27.19933149569036
        #FIXME: how many applications/averages
        self.num_repeats  = 250
        #FIXME: accuracy
        self.num_places   = 5

    def tearDown(self):
        self.A.Free()
        elem.Finalize()

    def test_apply_colwise(self):

        norm = 0.0
        #for _ in itertools.repeat(None, self.num_repeats):
        for i in range(self.num_repeats):

            #FIXME: how to choose seeds?
            #ctxt = cskylark.Context(random.randint(0, 1e9))
            ctxt = cskylark.Context(i)
            S    = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10, 6)
            SA   = elem.Mat()
            SA.Resize(6, 5)
            S.Apply(self.A, SA, 1)

            norm += np.linalg.norm(SA.Data())


        norm /= self.num_repeats

        self.assertAlmostEqual(norm, self.exp_col_norm, places=self.num_places)
        ctxt.Free()

    def test_apply_rowwise(self):

        norm = 0.0
        for i in range(self.num_repeats):

            #FIXME: how to choose seeds?
            ctxt = cskylark.Context(i)
            S    = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 5, 3)
            SA   = elem.Mat()
            SA.Resize(10, 3)
            S.Apply(self.A, SA, 2)

            norm += np.linalg.norm(SA.Data())

        norm /= self.num_repeats

        self.assertAlmostEqual(norm, self.exp_row_norm, places=self.num_places)
        ctxt.Free()


if __name__ == '__main__':
    unittest.main()
