import unittest
import numpy as np
from mpi4py import MPI
import math

from helper.elemental_matrix import create_elemental_matrix
from skylark import cskylark

import elem

class CT_test(unittest.TestCase):

    def setUp(self):
        elem.Initialize()

        #FIXME: accuracy?
        self.num_repeats = 5
        self.accuracy    = 1

    def tearDown(self):
        elem.Finalize()

    def test_apply_colwise(self):

        norm = 0.0
        A = create_elemental_matrix(10000, 100)
        norm_exp = np.linalg.norm(A.Data())

        for i in range(self.num_repeats):

            #FIXME: how to choose seeds?
            ctxt = cskylark.Context(i)
            S  = cskylark.CT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10000, 1000, 1.0)
            SA = elem.Mat()
            SA.Resize(1000, 100)
            S.Apply(A, SA, 1)

            norm += np.linalg.norm(SA.Data())

        norm /= self.num_repeats

        self.assertLess(math.fabs(norm - norm_exp) / norm, self.accuracy)

        ctxt.Free()
        A.Free()

    def test_apply_rowwise(self):

        norm = 0.0
        A = create_elemental_matrix(100, 10000)
        norm_exp = np.linalg.norm(A.Data())

        for i in range(self.num_repeats):

            #FIXME: how to choose seeds?
            ctxt = cskylark.Context(i)
            S  = cskylark.CT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10000, 1000, 1.0)
            SA = elem.Mat()
            SA.Resize(100, 1000)
            S.Apply(A, SA, 2)

            norm += np.linalg.norm(SA.Data())

        norm /= self.num_repeats

        self.assertLess(math.fabs(norm - norm_exp) / norm, self.accuracy)

        ctxt.Free()
        A.Free()

if __name__ == '__main__':
    unittest.main()
