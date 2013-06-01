import unittest
import numpy as np
from mpi4py import MPI

#TODO: move common helpers to a more suitable directory
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../../functionality'))
from helper.elemental_matrix import create_elemental_matrix

from helper.perftest import perftest
from skylark import cskylark

import elem

class CWT_test(unittest.TestCase):

    def setUp(self):
        # Parameters
        self.m  = 2000
        self.n  = 500
        self.sm = 200
        self.sn = 20

        # Create a matrix to sketch
        elem.Initialize()
        self.A = create_elemental_matrix(self.m, self.n)

        # Initilize context
        self.ctxt = cskylark.Context(123834)


    def tearDown(self):
        self.A.Free()
        self.ctxt.Free()
        elem.Finalize()


    @perftest
    def test_apply_colwise(self):
        S  = cskylark.CWT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", self.m, self.sm)
        SA = elem.Mat()
        SA.Resize(self.sm, self.n)
        S.Apply(self.A, SA, 1)

        #TODO: fail test if performance drops by a large factor


    @perftest
    def test_apply_rowwise(self):
        S  = cskylark.CWT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", self.n, self.sn)
        SA = elem.Mat()
        SA.Resize(self.m, self.sn)
        S.Apply(self.A, SA, 2)

        #TODO: fail test if performance drops by a large factor



if __name__ == '__main__':
    unittest.main()
