import unittest
import numpy as np
from mpi4py import MPI

#TODO: move common helpers to a more suitable directory
import os,sys
path = "%s" % (sys.path[0])
path.rstrip('/')
path = path[:path.rfind('/')]
sys.path.append(path + '/functionality/helper')
from elemental_matrix import create_elemental_matrix

from helper.perftest import perftest
from skylark import cskylark

import elem

class FJLT_test(unittest.TestCase):

    def setUp(self):
        # Create a matrix to sketch
        elem.Initialize()

        # Initilize context
        self.ctxt = cskylark.Context(123834)
        self.n    = 100000
        self.sn   = 1000


    def tearDown(self):
        self.ctxt.Free()
        elem.Finalize()


    @perftest
    def test_apply_colwise(self):
        A  = create_elemental_matrix(self.n, 100)
        S  = cskylark.FJLT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", self.n, self.sn)
        SA = elem.Mat()
        SA.Resize(self.sn, 100)
        S.Apply(A, SA, 1)

        #TODO: fail test if performance drops by a large factor
        A.Free()


    @perftest
    def test_apply_rowwise(self):
        A  = create_elemental_matrix(100, self.n)
        S  = cskylark.FJLT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", self.n, self.sn)
        SA = elem.Mat()
        SA.Resize(100, self.sn)
        S.Apply(A, SA, 2)

        #TODO: fail test if performance drops by a large factor
        A.Free()



if __name__ == '__main__':
    unittest.main()
