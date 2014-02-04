import sys
import elem
import skylark, skylark.io, skylark.ml.utils, skylark.metrics
import numpy as np
import numpy.linalg
from mpi4py import MPI
from KernelMachine import *
import argparse
import cPickle

parser = argparse.ArgumentParser(description='Block Distributed ADMM Solver for Randomized Kernel Methods')

parser.add_argument("--testfile", type=str, help='Test dataset (file in libsvm format)', required=True)
parser.add_argument("--modelfile", type=str, help='Save model in filename', required=True)
parser.add_argument("--outputfile", type=str, help='Save predictions in filename', required=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
NumProcessors = comm.Get_size()

# load the model (as a dictionary)
f = open(args.modelfile,'rb')
model = cPickle.load(f)
f.close()

# Instantiate the model with the loaded dictionary
model = KernelMachine(**model)

# Load the data
objective = 0
if rank == 0:
    print "Parsing the data..."
    data = skylark.io.libsvm(args.testfile).read()

    # If missing features, then augment the data
    if data[0].shape[1] < model.RFTs[0].getindim():
        fulldim = model.RFTs[0].getindim()
        n = data[0].shape[0]
        partialdim = data[0].shape[1]
        data[0] = numpy.concatenate((data[0], numpy.zeros((n, fulldim - partialdim))))

shape_X = data[0].shape if rank == 0 else None
shape_X = comm.bcast(shape_X, root=0)
if rank == 0 :
    print "Distributing the matrix..."

# Get X, Y in VC,* distributed matrix, i.e. row distributed
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

# predict with the model
predictions, labels = model.predict(X.Matrix)

# need distributed accuracy computation
accuracy = skylark.metrics.classification_accuracy(labels, y[:,0])

print accuracy

# Write model to modelfile - need both coefficients as well as random number generator
