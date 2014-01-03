#!/usr/bin/env python
import skylark, skylark.sketch, skylark.cskylark
import elem
from mpi4py import MPI
comm = MPI.COMM_WORLD

m = 10
n = 5
k = 4
elem.Initialize()

grid = elem.Grid()
A = elem.DistMat_VR_STAR( grid )
A.Resize(m,n)

elem.UniformDistMat( A, m, n )

context = skylark.cskylark.Context(123836)

S = skylark.sketch.JLT(m, k, context)

SA = S.sketch(A, 1)

SA.Print("Sketched Matrix")

