#!/usr/bin/env python
# Usage: python sketch_solve.py libsvm_trainfile libsvm_testfile

import El
from mpi4py import MPI
import skylark.io
import skylark.sketch as sketch
import numpy as np
import urllib
import sys

D = 500
T = 10000;

RFT = sketch.GaussianRFT
RFT_PARAMS = { 'sigma' : 10 }
RPT = sketch.CWT

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# If on rank 0 read data 
if rank == 0:
    print "Loading training data..."
    trnfile = skylark.io.libsvm(sys.argv[1])
    data = trnfile.read()

# Now broadcast the sizes
shape_X = data[0].shape if rank == 0 else None
shape_X = MPI.COMM_WORLD.bcast(shape_X, root=0)

# Transfer to Elemental format and redistribute the matrix and labels
if rank == 0 : print "Distributing the matrix..."
X_cc = El.DistMatrix(colDist = El.CIRC, rowDist = El.CIRC)
El.Zeros(X_cc, shape_X[0], shape_X[1])
Y_cc = El.DistMatrix(colDist = El.CIRC, rowDist = El.CIRC)
El.Zeros(Y_cc, shape_X[0], 1)
if rank == 0:
    X_ll = X_cc.Matrix()
    s = 0
    row = 0
    for e in data[0].indptr[1:]:
        for idx in range(s, e):
            col = data[0].indices[idx]
            val = data[0].data[idx]
            X_ll.Set(row, col, val)
        s = e
        row = row + 1
    
    Y_ll = Y_cc.Matrix()
    for j in range(shape_X[0]):
        Y_ll.Set(j, 0, data[1][j] - 1)
    
X = El.DistMatrix(colDist = El.VC, rowDist = El.STAR)
El.Copy(X_cc, X)
Y = El.DistMatrix(colDist = El.VC, rowDist = El.STAR)
El.Copy(Y_cc, Y);

if rank == 0 : print "Doing the regression..."

n = X.Height()
d = X.Width()

# Create right-hand side for the regression
k = int(El.Max(Y)[0] + 1)
rY = El.DistMatrix(colDist = El.VC, rowDist = El.STAR)
rY.Resize(Y.Height(), k)
El.Fill(rY, -1.0);
for i in range(rY.LocalHeight()): rY.SetLocal(i, int(Y.GetLocal(i, 0)), 1.0)

# (in the last line we take advantage of the fact that 
# the local matrices of X, Y and RY correspond to the same row.

# Apply random feature transform
R = RFT(d, D, **RFT_PARAMS)
XR = R / X    # <-------- Apply on the rows

# Reduce number of rows by sketching on the colums
S = RPT(n, T, defouttype = "RootMatrix")
SXR = S * XR
SY = S * rY

# Solve the regression

# Solve the regression: SXR and SY reside on rank zero, so solving the equation 
# is done there.
if (rank == 0):
    # Solve using NumPy
    [W, res, rk, s] = np.linalg.lstsq(SXR.Matrix().ToNumPy(), SY.Matrix().ToNumPy())
else:
    W = None

# Test the solution -- we need to disribute the solution since 
# we need to apply R in a distributed manner (this should change in future
# versions of Skylark)

# Note: instead we could have sketched to [STAR, STAR] and did local solves

# Distribute the solution 
W = MPI.COMM_WORLD.bcast(W, root=0) 

# If on rank 0 read test set from the web
if rank == 0:
    print "Loading test data..."
    tstfile = skylark.io.libsvm(sys.argv[1])
    data = tstfile.read()

# Now broadcast the sizes
shape_Xt = data[0].shape if rank == 0 else None
shape_Xt = comm.bcast(shape_Xt, root=0)

# Redistribute the matrix and labels
if rank == 0: print "Distributing the test matrix..."
Xt_cc = El.DistMatrix(colDist = El.CIRC, rowDist = El.CIRC)
El.Zeros(Xt_cc, shape_Xt[0], shape_Xt[1])
Yt_cc = El.DistMatrix(colDist = El.CIRC, rowDist = El.CIRC)
El.Zeros(Yt_cc, shape_Xt[0], 1)
if rank == 0:
    Xt_ll = Xt_cc.Matrix()
    s = 0
    row = 0
    for e in data[0].indptr[1:]:
        for idx in range(s, e):
            col = data[0].indices[idx]
            val = data[0].data[idx]
            Xt_ll.Set(row, col, val)
        s = e
        row = row + 1
    
    Yt_ll = Yt_cc.Matrix()
    for j in range(shape_Xt[0]):
        Yt_ll.Set(j, 0, data[1][j] - 1)
    
Xt = El.DistMatrix(colDist = El.VC, rowDist = El.STAR)
El.Copy(Xt_cc, Xt)
Yt = El.DistMatrix(colDist = El.VC, rowDist = El.STAR)
El.Copy(Yt_cc, Yt);

# Apply random features to Xt
if rank == 0: print "Doing the prediciton..."
XtR = R / Xt

# Elemental does not have a DistMatrix * LocalMatrix operation.
# But luckly for [VC, STAR] this operation is very simple
Yp_local = np.dot(XtR.Matrix().ToNumPy(), W).argmax(axis=1)

# Now evaluate solution
correct = np.equal(Yp_local, Yt.Matrix().ToNumPy().flatten()).sum()
correct = comm.reduce(correct)
if rank == 0:
    accuracy = (100.0 * correct) / Xt.Height()
    print "Accuracy rate is %.2f%%" % accuracy



