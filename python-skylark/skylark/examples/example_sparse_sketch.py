#!/usr/bin/python
from mpi4py import MPI
from skylark import cskylark

#TODO: use Mat.py and Vec.py (provided by KDT)?
import pyCombBLAS

import math


def print_sparse_mat(mat):
  print("Sketched A (CWT sparse, columnwise)")
  reti = pyCombBLAS.pyDenseParVec(30, 0)
  retj = pyCombBLAS.pyDenseParVec(30, 0)
  retv = pyCombBLAS.pyDenseParVec(30, 0)
  mat.Find(reti, retj, retv)
  for i in range(0, len(reti)):
    #TODO: why do we have 0.0 in output?
    if retv[i] != 0.0:
      print "(" + str(int(reti[i])) + ", " + str(int(retj[i])) + ") = " + str(retv[i])


ctxt = cskylark.Context(123836)
S    = cskylark.CWT(ctxt, "DistSparseMatrix", "DistSparseMatrix", 10, 6)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# creating an example dense matrix
rows = pyCombBLAS.pyDenseParVec(60, 0)
cols = pyCombBLAS.pyDenseParVec(60, 0)
vals = pyCombBLAS.pyDenseParVec(60, 0.0)
for i in range(0, 60):
  rows[i] = math.floor(i / 6)
for i in range(0, 60):
  cols[i] = i % 6
for i in range(0, 60):
  vals[i] = i

ACB  = pyCombBLAS.pySpParMat(10, 6, rows, cols, vals)

nullVec = pyCombBLAS.pyDenseParVec(0, 0)
SACB    = pyCombBLAS.pySpParMat(6, 6, nullVec, nullVec, nullVec)

S.Apply(ACB, SACB, 1)

if (rank == 0):
  print("Sketched A (CWT sparse, columnwise)")
  print_sparse_mat(SACB)

S.Free()

S       = cskylark.CWT(ctxt, "DistSparseMatrix", "DistSparseMatrix", 6, 3)
SACB    = pyCombBLAS.pySpParMat(10, 3, nullVec, nullVec, nullVec)
S.Apply(ACB, SACB, 2)

if (rank == 0):
  print("Sketched A (CWT sparse, columnwise)")
  print_sparse_mat(SACB)


S.Free()
ctxt.Free()
