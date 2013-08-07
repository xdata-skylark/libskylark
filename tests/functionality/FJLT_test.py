import unittest
import numpy as np
from mpi4py import MPI
import math

from skylark import cskylark

import elem

_M = 10000
_N = 100
_T = 1000

class FJLT_test(unittest.TestCase):

    def setUp(self):
        # We do not initialize Skylark so it will take the seed
        # from system time (new seed each time).
        # To be clean, and have every test self contained, we reinitalize
        # Skylark with a new time based seed
        cskylark.initialize()

        self.num_repeats = 5
        self.accuracy    = 1e-2

    def tearDown(self):
        # No real need to do this...
        cskylark.finalize()

    def test_apply_colwise(self):
        norm = 0.0
        A = elem.DistMatrix_d_VR_STAR()
        elem.Uniform(A, _M, _N)

        # To compute norm we have to go through (MC, MR) norm...
        A1 = elem.DistMatrix_d()
        elem.Copy(A, A1)
        norm_exp = elem.FrobeniusNorm(A1)

        for i in range(self.num_repeats):
            S  = cskylark.FJLT(_M, _T, intype="DistMatrix_VR_STAR")
            SA = np.zeros((_T, _N), order='F')
            S.apply(A, SA, "columnwise")
            SA = MPI.COMM_WORLD.bcast(SA, root=0)
            norm += np.linalg.norm(SA)

        norm /= self.num_repeats

        self.assertLess(math.fabs(norm - norm_exp) / norm, self.accuracy)

    def test_apply_rowwise(self):
        norm = 0.0
        A = elem.DistMatrix_d_VR_STAR()
        elem.Uniform(A, _N, _M)

        # To compute norm we have to go through (MC, MR) norm...
        A1 = elem.DistMatrix_d()
        elem.Copy(A, A1)
        norm_exp = elem.FrobeniusNorm(A1)

        for i in range(self.num_repeats):
            S  = cskylark.FJLT(_M, _T, intype="DistMatrix_VR_STAR")
            SA = np.zeros((_N, _T), order='F')
            S.apply(A, SA, "rowwise")
            SA = MPI.COMM_WORLD.bcast(SA, root=0)
            norm += np.linalg.norm(SA)

        norm /= self.num_repeats

        self.assertLess(math.fabs(norm - norm_exp) / norm, self.accuracy)

if __name__ == '__main__':
    unittest.main()
