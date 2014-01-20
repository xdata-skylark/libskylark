#!/usr/bin/env python
from mpi4py import MPI
import elem
import skylark.io
import skylark.sketch as sketch
import skylark.elemhelper
import numpy as np
import urllib
import bz2

DATASET_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2'
TESTDATA_URL = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2'

SIGMA = 10.35;
D = 500;
T = 10000;

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# If on rank 0 read data from the web
if rank == 0:
    print "Downloading dataset from web..."
    urlfile = urllib.urlopen(DATASET_URL)
    cmprs_data = urlfile.read()
    print "Decompressing the data..."
    rawdata = bz2.decompress(cmprs_data)
    print "Parsing the data..."
    data = skylark.io.sparselibsvm2scipy(rawdata.splitlines())
    urlfile.close()

# Now broadcast the sizes
shape_X = data[0].shape if rank == 0 else None
shape_X = MPI.COMM_WORLD.bcast(shape_X, root=0)

# Redistribute the matrix and labels
if rank == 0 : print "Distributing the matrix..."
X_cc = elem.DistMatrix_d_CIRC_CIRC(shape_X[0], shape_X[1])
Y_cc = elem.DistMatrix_d_CIRC_CIRC(shape_X[0], 1)
if rank == 0:
    X_cc.Matrix[:] = data[0].todense()
    data[1].resize((shape_X[0], 1))
    np.copyto(Y_cc.Matrix, data[1])
X = elem.DistMatrix_d_VC_STAR()
elem.Copy(X_cc, X);
Y = elem.DistMatrix_d_VC_STAR()
elem.Copy(Y_cc, Y);

n = X.Height
d = X.Width

# Create right-hand side for the regression
k = Y.Matrix.max() + 1
k = int(MPI.COMM_WORLD.allreduce(k, None, op=MPI.MAX))
rY = elem.DistMatrix_d_VC_STAR(n, k)
rY.Matrix.fill(-1.0)
for i in range(rY.LocalHeight): rY.Matrix[i, Y.Matrix[i, 0]] = 1.0

# (in the last line we take advantage of the fact that 
# the local matrices of X, Y and RY correspond to the same row.

# Apply random feature transform
R = sketch.GaussianRFT(d, D, SIGMA, defouttype="DistMatrix_VC_STAR")
XR = R / X    # <-------- Apply on the rows

# Reduce number of rows by sketching on the colums
S = sketch.CWT(n, T)
SXR = S * XR
SY = S * rY

# Solve the regression

# Solve the regression: SXR and SY reside on rank zero, so solving the equation 
# is done there.
if (rank == 0):
    # Solve using NumPy
    [W, res, rk, s] = np.linalg.lstsq(SXR, SY)
else:
    W = None

# Test the solution -- we need to disribute the solution since 
# we need to apply R in a distributed manner (this should change in future
# versions of Skylark)

# Distribute the solution 
W = MPI.COMM_WORLD.bcast(W, root=0) 

# If on rank 0 read test set from the web
if rank == 0:
    print "Downloading test datfrom web..."
    urlfile = urllib.urlopen(TESTDATA_URL)
    cmprs_data = urlfile.read()
    print "Decompressing the data..."
    rawdata = bz2.decompress(cmprs_data)
    print "Parsing the data..."
    data = skylark.io.sparselibsvm2scipy(rawdata.splitlines(), shape_X[1])
    urlfile.close()

# Now broadcast the sizes
shape_Xt = data[0].shape if rank == 0 else None
shape_Xt = comm.bcast(shape_Xt, root=0)

# Redistribute the matrix and labels
if rank == 0: print "Distributing the test matrix..."
Xt_cc = elem.DistMatrix_d_CIRC_CIRC(shape_Xt[0], shape_Xt[1])
Yt_cc = elem.DistMatrix_d_CIRC_CIRC(shape_Xt[0], 1)
if rank == 0:
    Xt_cc.Matrix[:] = data[0].todense()
    data[1].resize((shape_Xt[0], 1))
    np.copyto(Yt_cc.Matrix, data[1])
Xt = elem.DistMatrix_d_VC_STAR()
elem.Copy(Xt_cc, Xt);
Yt = elem.DistMatrix_d_VC_STAR()
elem.Copy(Yt_cc, Yt);

# Apply random features to Xt
if rank == 0: print "Doing the prediciton..."
XtR = R / Xt

# Elemental does not have a DistMatrix * LocalMatrix operation.
# But luckly for [VC, STAR] this operation is very simple
Yp_local = np.dot(XtR.Matrix, W).argmax(axis=1)

# Now evaluate solution
correct = np.equal(Yp_local, Yt.Matrix.flatten()).sum()
correct = comm.reduce(correct)
if rank == 0:
    accuracy = (100.0 * correct) / Xt.Height
    print "Accuracy rate is %.2f%%" % accuracy



