#!/usr/bin/env python

# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

from mpi4py import MPI
from skylark import cskylark
import skylark.io
import kdt
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

libsvm_path = '../datasets/usps.t'
libsvm_store = skylark.io.libsvm(libsvm_path)
(libsvm_mat, labels) = libsvm_store.read()

mtx_path = './usps.mtx'
mtx_store = skylark.io.mtx(mtx_path) 
mtx_store.write(libsvm_mat)
combblas_mat = mtx_store.read('combblas-sparse')

nrows = combblas_mat.nrow()
ncols = combblas_mat.ncol()

if rank == 0:
  print "Read a %s x %s matrix" % (nrows, ncols)
comm.barrier()

