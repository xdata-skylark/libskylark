#!/usr/bin/env python
import h5py
import elem
import numpy
from mpi4py import MPI
import skylark, skylark.io

comm = MPI.COMM_WORLD

# create a HDF5 file.
filename = 'mydataset.hdf5'
   
# create a 5 x 10 dat matrix and dump into filename 
if comm.Get_rank() == 0:
        m = 8
        n = 10
        writer = skylark.io.hdf5(filename, "w")
        writer.write_dense(numpy.array(range(1,81)).reshape(m,n), dataset = 'MyDataset')
        writer.close()
        
# Let all processes wait till the file is created.
comm.barrier()
    
# All processes create a reader object associated with the file  and query it to get the matrix dimensions
reader = skylark.io.hdf5(filename, "r")
(m, n) = reader.dimensions("MyDataset")
    
# Prepare Elemental matrix of size (m,n) for reading.
elem.Initialize()
grid = elem.Grid()
A = elem.DistMat_VC_STAR( grid )
A.Resize(m, n) # allocates a buffer.
    
# read from the reader into A, the dataset "MyDataset"
# Note that a single HDF5 file can contain multiple datasets e.g., features and labels.
#if pypar.rank() == 0:
reader.read_dense(A, "MyDataset")
reader.close()
        
# Check that Elemental has the dataset - need to fix column-major / row-major issue somehow.
A.Print("A - final rank=%d" % grid.Rank())
    
elem.Finalize()
