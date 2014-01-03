#!/usr/bin/env python
from mpi4py import MPI
from skylark import cskylark
from skylark.io import sparselibsvm2combBLAS
import pyCombBLAS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

filename = '../datasets/usps.t'
(mat, labels) = sparselibsvm2combBLAS(filename)

nrows = mat.getnrow()
ncols = mat.getncol()
print "Read a %s x %s matrix" % (nrows, ncols)

S    = cskylark.CWT(nrows, 100)

nullVec = pyCombBLAS.pyDenseParVec(0, 0)
sketch  = pyCombBLAS.pySpParMat(100, ncols, nullVec, nullVec, nullVec)

S.apply(mat, sketch, "rowwise")

if (rank == 0):
  print("Sketched A (CWT sparse, columnwise)")

