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

class JLT_test(unittest.TestCase):

    def setUp(self):
        # Create a matrix to sketch
        elem.Initialize()

        # Initilize context
        self.ctxt = cskylark.Context(123834)


    def tearDown(self):
        self.ctxt.Free()
        elem.Finalize()


    @perftest
    def test_apply_colwise(self):
        A  = create_elemental_matrix(10000, 100)
        S  = cskylark.JLT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", 10000, 1000)
        SA = elem.Mat()
        SA.Resize(1000, 100)
        S.Apply(A, SA, 1)

        #TODO: fail test if performance drops by a large factor
        A.Free()


    @perftest
    def test_apply_rowwise(self):
        A  = create_elemental_matrix(100, 10000)
        S  = cskylark.JLT(self.ctxt, "DistMatrix_VR_STAR", "Matrix", 10000, 1000)
        SA = elem.Mat()
        SA.Resize(100, 1000)
        S.Apply(A, SA, 2)

        #TODO: fail test if performance drops by a large factor
        A.Free()



if __name__ == '__main__':
    unittest.main()
