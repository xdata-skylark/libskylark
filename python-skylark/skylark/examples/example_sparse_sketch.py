#!/usr/bin/python
from mpi4py import MPI
from skylark import cskylark

import kdt
import math

ctxt = cskylark.Context(123836)
S    = cskylark.CWT(ctxt, "DistSparseMatrix", "DistSparseMatrix", 10, 6)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# creating an example matrix (pySpParMat can only be created from dense vectors)
rows = kdt.Vec(60, sparse=False)
cols = kdt.Vec(60, sparse=False)
vals = kdt.Vec(60, sparse=False)
for i in range(0, 60):
  rows[i] = math.floor(i / 6)
for i in range(0, 60):
  cols[i] = i % 6
for i in range(0, 60):
  vals[i] = i

#XXX: insert values directly into matrix?
ACB = kdt.Mat(rows, cols, vals, 6, 10)
print ACB

nullVec = kdt.Vec(0, sparse=False)
SACB    = kdt.Mat(nullVec, nullVec, nullVec, 6, 6)

S.Apply(ACB, SACB, 1)

if (rank == 0):
  print("Sketched A (CWT sparse, columnwise)")
  print SACB

S.Free()

S       = cskylark.CWT(ctxt, "DistSparseMatrix", "DistSparseMatrix", 6, 3)
SACB    = kdt.Mat(nullVec, nullVec, nullVec, 3, 10)
S.Apply(ACB, SACB, 2)

if (rank == 0):
  print("Sketched A (CWT sparse, rowwise)")
  print SACB


S.Free()
ctxt.Free()
