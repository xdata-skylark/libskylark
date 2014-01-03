#!/usr/bin/env python
# MPI usage: 
# mpiexec -np 2 python skylark/examples/example_cskylark.py

import elem
from skylark import cskylark, elemhelper
from mpi4py import MPI
import numpy as np
import time

# Configuration 
m = 20000;
n = 300;
t = 1000;
sketches = { "JLT" : cskylark.JLT, "FJLT" : cskylark.FJLT, "CWT" : cskylark.CWT }

# Set up the random regression problem.
A = elem.DistMatrix_d_VR_STAR()
elem.Uniform(A, m, n)
b = elem.DistMatrix_d_VR_STAR()
elem.Uniform(b, m, 1)

# Solve using Elemental
# Elemental currently does not support LS on VR,STAR.
# So we copy.
A1 = elem.DistMatrix_d()
elem.Copy(A, A1)
b1 = elem.DistMatrix_d()
elem.Copy(b, b1)
x = elem.DistMatrix_d(n, 1)
t0 = time.time()
elem.LeastSquares(elem.NORMAL, A1, b1, x)
telp = time.time() - t0

# Compute residual
r = elem.DistMatrix_d()
elem.Copy(b, r)
elem.Gemv(elem.NORMAL, -1.0, A1, x, 1.0, r)
res = elem.Norm(r)
if (MPI.COMM_WORLD.Get_rank() == 0):
  print "Exact solution residual %(res).3f\t\t\ttook %(elp).2e sec" % \
      { "res" : res, "elp": telp }

# Skylark is automatically initilalized when you import Skylark,
# It will use system time to generate the seed. However, we can 
# reinitialize for so to fix the seed.
cskylark.initialize(123834);

#
# Solve the problem using sketching
#

for sname in sketches:
  stype = sketches[sname]
  
  t0 = time.time()

  # Create transform. Default output type is "LocalMatrix".
  S = stype(m, t)

  # Sketch both A and b using the same sketch
  SA = S * A
  Sb = S * b
  
  # SA and Sb reside on rank zero, so solving the equation is 
  # done there.
  if (MPI.COMM_WORLD.Get_rank() == 0):
    # Solve using NumPy
    [x, res, rank, s] = np.linalg.lstsq(SA, Sb)
  else:
    x = None

  telp = time.time() - t0

  # Distribute the solution so to compute residual in a distributed fashion
  x = MPI.COMM_WORLD.bcast(x, root=0)
    
  # Convert x to a distributed matrix.
  # Here we give the type explictly, but the value used is the default.
  x = elemhelper.local2distributed(x, type=elem.DistMatrix_d)

  # Compute residual
  r = elem.DistMatrix_d()
  elem.Copy(b, r)
  elem.Gemv(elem.NORMAL, -1.0, A1, x, 1.0, r)
  res = elem.Norm(r)
  if (MPI.COMM_WORLD.Get_rank() == 0):
    print "%(name)s:\tSketched solution residual %(val).3f\ttook %(elp).2e sec" %\
        {"name" : sname, "val" : res, "elp" : telp}

  # As with all Python object they will be automatically garbage
  # collected, and the associated memory will be freed.
  # You can also explicitly free them.
  del S     # S = 0 will also free memory.

# Really no need to close skylark -- it will do it automatically.
# However, if you really want to you can do it.
cskylark.finalize()


