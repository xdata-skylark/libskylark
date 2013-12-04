import unittest
import numpy as np

from mpi4py import MPI

from skylark import cskylark

from helper.test import svd_bound
from helper.test import test_helper

import elem

_M = 10000
_N = 100
_R = 1000

class SVD_test(unittest.TestCase):

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

        self.sketches = [cskylark.JLT, cskylark.FJLT, cskylark.CWT]

    def tearDown(self):
        # No real need to do this...
        #cskylark.finalize()
        pass

    def check_result(self, results, sketch_name):
        suc = np.zeros(len(results[0].success))
        avg = np.zeros(len(results[0].average))
        for result in results:
            avg = avg + result.average
            suc = np.logical_or(suc, result.success)

        # check if at leaste one was successful
        self.assertTrue(np.all(suc), msg=("Failed checking " + sketch_name))

        # check if average is in bounds
        avg = avg / self.num_repeats
        self.assertTrue(np.all(avg <= self.accuracy),
                        msg=("Failed checking " + sketch_name))

    def test_apply_colwise(self):
        A = elem.DistMatrix_d_VR_STAR()

        #FIXME: Christos, use your matrix problem factory here
        elem.Uniform(A, _M, _N)

        for sketch in self.sketches:
            results = test_helper(A, _M, _N, _R, sketch, [svd_bound], MPI)
            self.check_result(results, str(sketch))

    def test_apply_rowwise(self):
        A = elem.DistMatrix_d_VR_STAR()

        #FIXME: Christos, use your matrix problem factory here
        elem.Uniform(A, _M, _N)

        #measures = [svd_bound] #.. add more measures to be computed in a test
        #results  = test_helper(A, _M, _N, _R, cskylark.JLT, measures, MPI, direction="rowwise")

        #self.check_result(results)


if __name__ == '__main__':
    unittest.main()
