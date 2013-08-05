#!/usr/bin/python
# Usage: 
# mpiexec -np 2 python ../mypython_packages/lib/python2.7/site-packages/skylark/examples/example_cskylark.py

import elem
from skylark import cskylark
from mpi4py import MPI

# Create a matrix to sketch
A = elem.DistMatrix_d_VR_STAR(10, 5)
localHeight = A.LocalHeight
localWidth = A.LocalWidth
colShift = A.ColShift
rowShift = A.RowShift
colStride = A.ColStride
rowStride = A.RowStride
data = A.Matrix
ldim = A.LDim
for jLocal in xrange(0,localWidth):
  j = rowShift + jLocal*rowStride
  for iLocal in xrange(0,localHeight):
    i = colShift + iLocal*colStride
    data[iLocal, jLocal] = i-j
elem.Display(A, "Original A");

# Initilize context
ctxt = cskylark.Context(123834)

# Create JLT transform
S = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10, 6)

# Apply it
SA = elem.Matrix_d(6, 5)
S.Apply(A, SA, 1)
if (MPI.COMM_WORLD.Get_rank() == 0):
  SA.Print("Sketched A (JLT)")

# Repeat with FJLT
T = cskylark.FJLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10, 6)
TA = elem.Mat()
TA.Resize(6, 5)
T.Apply(A, TA, 1)
if (MPI.COMM_WORLD.Get_rank() == 0):
  TA.Print("Sketched A (FJLT)")

# Clean up
S.Free()
T.Free()
ctxt.Free()

TA.Free()
SA.Free()
A.Free()

elem.Finalize()
