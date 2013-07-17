#!/usr/bin/python
# Usage: 
# mpiexec -np 2 python ../mypython_packages/lib/python2.7/site-packages/skylark/examples/example_cskylark.py

import elem
from skylark import cskylark
from mpi4py import MPI

elem.Initialize()

# Create a matrix to sketch
# TODO: non uniform function in elemental interface...
grid = elem.Grid()
A = elem.DistMat_VR_STAR( grid )
A.Resize(10,5)
localHeight = A.LocalHeight()
localWidth = A.LocalWidth()
colShift = A.ColShift()
rowShift = A.RowShift()
colStride = A.ColStride()
rowStride = A.RowStride()
data = A.Data()
ldim = A.LDim()
for jLocal in xrange(0,localWidth):
  j = rowShift + jLocal*rowStride
  for iLocal in xrange(0,localHeight):
    i = colShift + iLocal*colStride
    data[iLocal+jLocal*ldim] = i-j
A.Print("Original A")

# Initilize context
ctxt = cskylark.Context(123834)

# Create JLT transform
S = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "Matrix", 10, 6)

# Apply it
SA = elem.Mat()
SA.Resize(6, 5)
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
