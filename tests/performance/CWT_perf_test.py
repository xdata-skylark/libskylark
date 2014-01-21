import elem
import numpy as np

from helper.perftest import dump_timings
from helper.perftest import TestCase
from skylark import sketch

class CWT_test(TestCase):

    def setUp(self):
        sketch.initialize()

        self.n    = 100000
        self.sn   = 1000

    def tearDown(self):
        pass

    @dump_timings
    def test_apply_colwise(self):
        A = elem.DistMatrix_d_VR_STAR()
        elem.Uniform(A, self.n, 100)

        S  = sketch.CWT(self.n, self.sn)
        SA = np.zeros((self.sn, 100), order='F')
        S.apply(A, SA, "columnwise")

    @dump_timings
    def test_apply_rowwise(self):
        A = elem.DistMatrix_d_VR_STAR()
        elem.Uniform(A, 100, self.n)

        S  = sketch.CWT(self.n, self.sn)
        SA = np.zeros((100, self.sn), order='F')
        S.apply(A, SA, "rowwise")


if __name__ == '__main__':
    perftest = CWT_test()
    perftest.main()
