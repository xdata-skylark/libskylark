#!/usr/bin/env python
# Usage: python svd.py

import El
from mpi4py import MPI
from skylark.nla import svd as randsvd
import numpy as np
import sys

k = 10

A = El.DistMatrix()

print "Computing El.SVD:"
El.Demmel(A, 100)
(S_el, _) = El.SVD(A)
print S_el.Matrix().ToNumPy()[:k]


print "Computing randSVD with k=20:"
El.Demmel(A, 100)
parms = randsvd.Params()
parms.num_iterations = 10
(U, S, V) = randsvd.approximate_svd(A, k, parms)
print S.Matrix().ToNumPy()

print np.linalg.norm((S.Matrix().ToNumPy() - S_el.Matrix().ToNumPy()[:k]), ord='fro')

