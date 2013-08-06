#!/usr/bin/python
# Usage: 
# mpiexec -np 2 python ../mypython_packages/lib/python2.7/site-packages/skylark/examples/example_cskylark.py

import elem
from skylark import cskylark
from mpi4py import MPI
import numpy as np

# Create a matrix to sketch
A = elem.DistMatrix_d_VR_STAR()
elem.Uniform(A, 10, 5);
elem.Display(A, "Original A");

# Initilize context
ctxt = cskylark.Context(123834)

# Create JLT transform
S = cskylark.JLT(ctxt, "DistMatrix_VR_STAR", "LocalMatrix", 10, 6)

# Apply it
SA = np.zeros((6, 5), order='F')
S.apply(A, SA, 1)
if (MPI.COMM_WORLD.Get_rank() == 0):
  print "Sketched A (JLT)"
  print SA;

# Repeat with FJLT
T = cskylark.FJLT(ctxt, "DistMatrix_VR_STAR", "LocalMatrix", 10, 6)
TA = np.zeros((6, 5), order='F')
T.apply(A, TA, 1)
if (MPI.COMM_WORLD.Get_rank() == 0):
  print "Sketched A (FJLT)"
  print TA;

# Clean up
S.free()
T.free()
ctxt.free()

