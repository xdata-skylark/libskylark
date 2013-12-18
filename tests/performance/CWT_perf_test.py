import numpy as np

from helper.perftest import dump_timings
from helper.perftest import TestCase
from skylark import cskylark

import elem

class CWT_test(TestCase):

    def setUp(self):
        cskylark.initialize()

        self.n    = 100000
        self.sn   = 1000


    def tearDown(self):
        pass


    @dump_timings
    def test_apply_colwise(self):
        A = elem.DistMatrix_d_VR_STAR()
        elem.Uniform(A, self.n, 100)

        S  = cskylark.CWT(self.n, self.sn, intype="DistMatrix_VR_STAR")
        SA = np.zeros((self.sn, 100), order='F')
        S.apply(A, SA, "columnwise")

        #TODO: fail test if performance drops by a large factor


    @dump_timings
    def test_apply_rowwise(self):
        A = elem.DistMatrix_d_VR_STAR()
        elem.Uniform(A, 100, self.n)

        S  = cskylark.CWT(self.n, self.sn, intype="DistMatrix_VR_STAR")
        SA = np.zeros((100, self.sn), order='F')
        S.apply(A, SA, "rowwise")

        #TODO: fail test if performance drops by a large factor



if __name__ == '__main__':
    perftest = CWT_test()
    perftest.main()
