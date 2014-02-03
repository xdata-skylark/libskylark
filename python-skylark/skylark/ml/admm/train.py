import sys
import elem
import skylark, skylark.io, skylark.ml.utils
import numpy as np
import numpy.linalg
from mpi4py import MPI
from KernelMachine import *
import argparse
from proxlibrary import loss
import cProfile, pstats, StringIO

parser = argparse.ArgumentParser(description='Block Distributed ADMM Solver for Randomized Kernel Methods')

parser.add_argument("--trainfile", type=str, help='Training dataset (file in libsvm format)', required=True)

# if its a squared loss based model, we need to know how to treat the labels -- TODO
parser.add_argument("--problem", type=str, help='Problem type (default: binary_classification|multiclass_classification|regression)', default='multiclass_classification')

parser.add_argument("--lossfunction", type=str, help='Loss function (squared, crossentropy, hinge - default: squared)', default='squared')
parser.add_argument("--regularizer", type=str, help='Regularizer (l2,l1 - default: l2)', default='l2')

parser.add_argument("--kernel", type=str, help='Kernel (default: gaussian)', default='gaussian')
parser.add_argument("--kernelparam", type=float, help='Kernel Parameters (e.g., Gaussian Kernel bandwidth) - negative value means estimate', required=True)

parser.add_argument("--randomfeatures", type=int, help='Number of random features (default: 1000)', default=1000)
parser.add_argument("--regparam", type=float, help='Regularization parameter (i.e. lambda - default: 0)', default=0)

parser.add_argument("--numfeaturepartitions", type=float, help='Number of Feature partitions (default: 5)', default=5)

parser.add_argument("--TOL", type=float, help='Convergance tolerance (default: 1e-3)', default=0.001)
parser.add_argument("--MAXITER", type=int, help='Maximum ADMM iterations (default: 10)', default=10)
parser.add_argument("--SEED", type=int, help='Seed for random numbers (default: 12345)', default=12345)
parser.add_argument("--fastfood", help='Enable Fastfood acceleration', action='store_true')

parser.add_argument("--modelfile", type=str, help='Save model in filename', required=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
NumProcessors = comm.Get_size()

# Load the data
objective = 0
if rank == 0:
    print "Parsing the data..."
    starttime = MPI.Wtime()
    data = skylark.io.sparselibsvm2scipy(args.trainfile)
    print 'Reading took %f seconds' % (MPI.Wtime() - starttime)
    x,y = data

    if args.lossfunction=="squared" or args.lossfunction=="lad":
        y = skylark.ml.utils.dummycoding(y)
        y = 2*y-1
    else:
        y = y - 1 # 0-to-K-1

    #n,d = x.shape
    #T = ExplicitFeatureMap(dimensions=d,kernelparam=args.kernelparam, randomfeatures=args.randomfeatures)
    #z = T.map(x.todense())

    #lossfn = loss(args.lossfunction)


    #objective = lambda W:  lossfn(numpy.dot(z,W), y) + 0.5*args.regparam*np.linalg.norm(W, 'fro')**2



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


#pr = cProfile.Profile()
if rank==0:
    print "Reading and distributing the data toolk %f seconds" % (MPI.Wtime() - starttime)

# train the model
model = KernelMachine(lossfunction=args.lossfunction,
                      regularizer=args.regularizer,
                      regparam=args.regparam,
                      randomfeatures=args.randomfeatures,
                      kernel=args.kernel,
                      kernelparam=args.kernelparam,
                      numfeaturepartitions=args.numfeaturepartitions,
                      TOL=args.TOL,
                      MAXITER=args.MAXITER,
                      SEED=args.SEED,
                      fastfood=args.fastfood)

#pr.enable()

model.train((X,Y))


#pr.disable()
#s = StringIO.StringIO()
#sortby = 'cumulative'
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#print s.getvalue()


# Write model to modelfile - need both coefficients as well as random number generator
model.save(args.modelfile)
