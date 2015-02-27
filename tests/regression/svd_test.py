import unittest
import numpy as np

from mpi4py import MPI

from skylark import sketch

from helper.test import test_helper
from collections import namedtuple

import El

_M = 10000
_N = 100
_R = 1000

def construct_CT(n, s, C=1.0):
    return sketch.CT(n, s, C)

def construct_WZT(n, s, C=2.0):
    return sketch.WZT(n, s, C)


class SVD_test(unittest.TestCase):

    def setUp(self):
        # We do not initialize Skylark so it will take the seed
        # from system time (new seed each time).
        # To be clean, and have every test self contained, we reinitalize
        # Skylark with a new time based seed
        sketch.initialize()

        #params
        self.num_repeats = 5
        self.accuracy    = 0.5
        _R = _N / self.accuracy**2

        #self.sketches = [sketch.JLT, sketch.FJLT, sketch.CWT]
        self.sketches = [sketch.JLT, sketch.CWT]

    def tearDown(self):
        pass


    def svd_bound(self, SA):
        """
        Test if the singular values of the original (M x N) and sketched matrix
        (R x N) are bounded by:

            SVD(A)_i * (1 - accuracy) <= SVD(SA)_i <= SVD(A)_i * (1 + accuracy)

        Computes the average relative error per index, and additionally returns
        a boolean vector describing, for each index, if we have at least one
        singular value honoring the bounds.
        """

        result = namedtuple('svd_bound_result', ['average', 'success'])

        sav = np.linalg.svd(SA, full_matrices=1, compute_uv=0)

        average = abs(self.sv - sav) / self.sv
        success = np.zeros(len(self.sv))

        for idx in range(len(self.sv)):
            success[idx] = self.sv[idx] * (1 - self.accuracy) <= sav[idx] <= self.sv[idx] * (1 + self.accuracy)

        return result._make([average, success])


    def check_result(self, results, sketch_name):
        suc = np.zeros(len(results[0].success))
        avg = np.zeros(len(results[0].average))
        for result in results:
            avg = avg + result.average
            suc = np.logical_or(suc, result.success)

        # check if at leaste one was successful
        error_str = "Some indicies not in bounds after %i repetitions of sketch %s" % (self.num_repeats, sketch_name)
        self.assertTrue(np.all(suc), msg=error_str)

        # check if average is in bounds
        avg = avg / self.num_repeats
        error_str = "Average error (rms) not in bounds of errors after %i reptitions of sketch %s" % (self.num_repeats, sketch_name)
        self.assertTrue(np.all(avg <= self.accuracy), msg=error_str)


    def test_apply_colwise(self):
        A = El.DistMatrix(El.dTag, El.VR, El.STAR)

        #FIXME: Christos, use your matrix problem factory here
        El.Uniform(A, _M, _N)

        #FIXME: A.Matrix will not work in parallel
        self.sv  = np.linalg.svd(A.Matrix().ToNumPy(), full_matrices=1, compute_uv=0)

        for sketch in self.sketches:
            results = test_helper(A, _M, _N, _R, sketch, [self.svd_bound], MPI)
            self.check_result(results, str(sketch))


    def test_apply_rowwise(self):
        A = El.DistMatrix(El.dTag, El.VR, El.STAR)

        #FIXME: Christos, use your matrix problem factory here
        El.Uniform(A, _N, _M)

        #FIXME: A.Matrix will not work in parallel
        self.sv  = np.linalg.svd(A.Matrix().ToNumPy(), full_matrices=1, compute_uv=0)

        for sketch in self.sketches:
            results = test_helper(A, _N, _M, _R, sketch, [self.svd_bound], MPI, 5, "rowwise")
            self.check_result(results, str(sketch))


if __name__ == '__main__':
    unittest.main(verbosity=2)
