import mpi4py.rc
mpi4py.rc.finalize = False

import numpy
import scipy
import elem
from mpi4py import MPI

from skylark import io

import unittest


class IO_test(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD

        self.ele_A = elem.DistMatrix_d()
        elem.Uniform(self.ele_A, 10, 30)

        self.np_A = numpy.random.random((20, 65))

        self.sp_A = scipy.sparse.rand(20, 65, density=0.2, format='csr')

    def tearDown(self):
        pass

    def compareMatrixNorm(self, A, B):
        # Gather at root
        A_CIRC_CIRC = elem.DistMatrix_d_CIRC_CIRC()
        elem.Copy(A, A_CIRC_CIRC)

        # Compare Frobenius norm
        self.assertAlmostEqual(
            numpy.linalg.norm(A_CIRC_CIRC.Matrix[:], ord='fro'),
            #FIXME: only to 5 places??
            numpy.linalg.norm(B, ord='fro'), 5)

    def test_mm(self):
        matrix_fpath = 'test_matrix.mtx'
        store = io.mtx(matrix_fpath)

        store.write(self.sp_A)
        B = store.read('combblas-sparse')
        C = store.read('scipy-sparse')
        D = store.read('numpy-dense')

        # convert CombBLAS matrix to (coo) sparse matrix
        col, row, data = B.toVec()
        sp_cb = scipy.sparse.coo_matrix((data, (row, col)), shape=(20, 65))
        self.assertTrue(((self.sp_A - sp_cb).todense() < 1e-7).all())

        self.assertTrue((self.sp_A.todense() - D < 1e-7).all())
        self.assertTrue(((self.sp_A - C).todense() < 1e-7).all())

        #FIXME: throws exception
        #store.write(B)
        #B = store.read('combblas-sparse')
        #C = store.read('scipy-sparse')
        #D = store.read('numpy-dense')

        ## convert CombBLAS matrix to (coo) sparse matrix
        #col, row, data = B.toVec()
        #sp_cb = scipy.sparse.coo_matrix((data, (row, col)), shape=(20, 65))
        #self.assertTrue(((self.sp_A - sp_cb).todense() < 1e-7).all())

        #self.assertTrue((self.sp_A.todense() - D < 1e-7).all())
        #self.assertTrue(((self.sp_A - C).todense() < 1e-7).all())

    def test_hdf5(self):
        fpath = 'test_matrix.h5'
        store = io.hdf5(fpath)
        store.write(self.np_A)

        B = store.read('numpy-dense')
        C = store.read('elemental-dense')
        D = store.read('elemental-dense', distribution='VC_STAR')

        self.compareMatrixNorm(C, self.np_A)
        self.compareMatrixNorm(D, self.np_A)
        #FIXME: why does this test fail?
        #self.compareMatrixNorm(C, B)

        store.write(self.ele_A)
        B = store.read('numpy-dense')
        C = store.read('elemental-dense')
        D = store.read('elemental-dense', distribution='VC_STAR')
        self.compareMatrixNorm(self.ele_A, B)
        self.compareMatrixNorm(C, B)
        self.compareMatrixNorm(D, B)

        with self.assertRaises(Exception):
            store.read('combblas-dense')

    def test_txt(self):
        fpath = 'test_matrix.txt'
        store = io.txt(fpath)

        store.write(self.ele_A)
        B = store.read('numpy-dense')
        self.compareMatrixNorm(self.ele_A, B)

        store.write(self.np_A)
        B = store.read('numpy-dense')
        self.assertTrue((self.np_A == B).all())

        with self.assertRaises(Exception):
            B = store.read('elemental-dense')
            B = store.read('combblas-dense')

    def test_libsvm(self):
        fpath = '../../python-skylark/skylark/datasets/usps.t'
        store = io.libsvm(fpath)
        features_matrix, labels_matrix = store.read()
        matrix_info = features_matrix.shape, features_matrix.nnz, labels_matrix.shape

        #FIXME: currently there is no way to test this in a sophisticated way.
        #       For now just test matrix_info
        self.assertEqual(matrix_info, ((2007, 257), 513792, (2007,)))


if __name__ == '__main__':
    unittest.main()
