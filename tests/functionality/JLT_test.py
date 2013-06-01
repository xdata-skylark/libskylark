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
        self.A = create_elemental_matrix(10000, 100)

        #FIXME: expected norm = ?
        self.exp_col_norm = 5767038.2619297039
        self.exp_row_norm = 27.19933149569036
        #FIXME: how many applications/averages
        self.num_repeats  = 5
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
            ctxt = cskylark.Context(random.randint(0, 1e9))
            #ctxt = cskylark.Context(i)
            S    = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10000, 1000)
            SA   = elem.Mat()
            SA.Resize(1000, 100)
            S.Apply(self.A, SA, 1)

            norm += np.linalg.norm(SA.Data())


        norm /= self.num_repeats

        #FIXME
        #self.assertAlmostEqual(norm, self.exp_col_norm, places=self.num_places)
        ctxt.Free()

    def test_apply_rowwise(self):

        norm = 0.0
        for i in range(self.num_repeats):

            #FIXME: how to choose seeds?
            ctxt = cskylark.Context(i)
            S    = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 100, 10)
            SA   = elem.Mat()
            SA.Resize(10000, 10)
            S.Apply(self.A, SA, 2)

            norm += np.linalg.norm(SA.Data())

        norm /= self.num_repeats

        #FIXME
        #self.assertAlmostEqual(norm, self.exp_row_norm, places=self.num_places)
        ctxt.Free()


if __name__ == '__main__':
    unittest.main()
