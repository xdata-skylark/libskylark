#!/usr/bin/env python
# MPI usage:
# mpiexec -np 2 python skylark/examples/example_sparse_sketch.py

# prevent mpi4py from calling MPI_Finalize()
import mpi4py.rc
mpi4py.rc.finalize   = False

from mpi4py import MPI
from skylark import sketch

import kdt
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Lower layers are automatically initilalized when you import Skylark,
# It will use system time to generate the seed. However, we can
# reinitialize for so to fix the seed.
sketch.initialize(123834);

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
S       = sketch.CWT(10, 6)
S.apply(ACB, SACB, "columnwise")

if (rank == 0):
  print("Sketched A (CWT sparse, columnwise)")
print SACB

# No need to free S -- it will be automatically garbage collected
# and the memory for the sketch reclaimed.

SACB = kdt.Mat(nullVec, nullVec, nullVec, 3, 10)
S    = sketch.CWT(6, 3)
S.apply(ACB, SACB, "rowwise")

if (rank == 0):
  print("Sketched A (CWT sparse, rowwise)")
print SACB

# Really no need to "close" the lower layers -- it will do it automatically.
# However, if you really want to you can do it.
sketch.finalize()
