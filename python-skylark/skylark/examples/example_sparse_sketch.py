#!/usr/bin/python

# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

from mpi4py import MPI
from skylark import cskylark

import kdt
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ctxt = cskylark.Context(123836)

# creating an example matrix
# It seems that pySpParMat can only be created from dense vectors
rows = kdt.Vec(60, sparse=False)
cols = kdt.Vec(60, sparse=False)
vals = kdt.Vec(60, sparse=False)
for i in range(0, 60):
  rows[i] = math.floor(i / 6)
for i in range(0, 60):
  cols[i] = i % 6
for i in range(0, 60):
  vals[i] = i

ACB = kdt.Mat(rows, cols, vals, 6, 10)
print ACB

nullVec = kdt.Vec(0, sparse=False)
SACB    = kdt.Mat(nullVec, nullVec, nullVec, 6, 6)
S       = cskylark.CWT(ctxt, "DistSparseMatrix", "DistSparseMatrix", 10, 6)
print type(ACB).__name__
S.apply(ACB, SACB, 1)

if (rank == 0):
  print("Sketched A (CWT sparse, columnwise)")
  print SACB

S.free()

SACB = kdt.Mat(nullVec, nullVec, nullVec, 3, 10)
S    = cskylark.CWT(ctxt, "DistSparseMatrix", "DistSparseMatrix", 6, 3)
S.apply(ACB, SACB, 2)

if (rank == 0):
  print("Sketched A (CWT sparse, rowwise)")
  print SACB

S.free()
ctxt.free()
