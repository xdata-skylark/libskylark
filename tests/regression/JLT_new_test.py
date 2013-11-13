import unittest
import numpy as np
from mpi4py import MPI
import math

from skylark import cskylark

import elem

_M = 10000
_N = 100
_R = 1000

class JLT_new_test(unittest.TestCase):

    def setUp(self):
        # We do not initialize Skylark so it will take the seed
        # from system time (new seed each time).
        # To be clean, and have every test self contained, we reinitalize
        # Skylark with a new time based seed
        cskylark.initialize()

        #params
        self.num_repeats = 5
        self.accuracy    = 0.5
        _R = _N / self.accuracy**2

    def tearDown(self):
        # No real need to do this...
        cskylark.finalize()

    def test_apply_colwise(self):
        A = elem.DistMatrix_d_VR_STAR()

        #FIXME: Christos, use your matrix problem factory here
        elem.Uniform(A, _M, _N)

        #FIXME: local
        sv = np.linalg.svd(A.Matrix, full_matrices=1, compute_uv=0)

        success = np.zeros(len(sv))
        average = np.zeros(len(sv))
        for i in range(self.num_repeats):
            S  = cskylark.JLT(_M, _R, intype="DistMatrix_VR_STAR")
            SA = np.zeros((_R, _N), order='F')
            S.apply(A, SA, "columnwise")
            SA = MPI.COMM_WORLD.bcast(SA, root=0)

            # check bounds of singular values
            sav = np.linalg.svd(SA, full_matrices=1, compute_uv=0)
            for idx in range(len(sv)):
                average[idx] += sv[idx]
                success[idx] = success[idx] or (sv[idx] * (1 - self.accuracy)
                    <= sav[idx] <= sv[idx] * (1 + self.accuracy))

        # check if at leaste one was successful
        self.assertTrue(np.all(success))

        # check if average is in bounds
        average = average / self.num_repeats
        for idx in range(len(average)):
            self.assertTrue(sv[idx] * (1 - self.accuracy) <= average[idx]
                    <= sv[idx] * (1 + self.accuracy))



if __name__ == '__main__':
    unittest.main()
