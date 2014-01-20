#!/usr/bin/env python

# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

import h5py
import elem
import numpy
import numpy.linalg
from mpi4py import MPI
import skylark, skylark.io

comm = MPI.COMM_WORLD

# Create a HDF5 file and encapsulate this and its metadata in store
filename = 'mydataset.hdf5' 
store = skylark.io.hdf5(filename, dataset='MyDataset')

# Create a 5 x 10 dat matrix and populate its store
if comm.Get_rank() == 0:
  m = 8
  n = 10
  matrix = numpy.array(range(1,81)).reshape(m,n)
  store.write(matrix)
      
# Let all processes wait till the file is created.
comm.barrier()
    
# All processes read into the file
A = store.read('elemental-dense', distribution='VC_STAR')

# Check the Frobenius norm of the difference of generated/written and read-back matrices
# Gather at root
A_CIRC_CIRC = elem.DistMatrix_d_CIRC_CIRC()
elem.Copy(A, A_CIRC_CIRC)

# Compute the norm at root and output
if comm.rank == 0:
  diff_fro_norm = numpy.linalg.norm(A_CIRC_CIRC.Matrix[:] - matrix, ord='fro')
  print '||generated_matrix - read_matrix||_F = %f' % diff_fro_norm 
