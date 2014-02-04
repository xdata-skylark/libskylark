from mpi4py import MPI
import numpy
from scipy import linalg
import cPickle
from proxlibrary import *
import elem
import sys
import math
import skylark.ml.utils
import json
import scipy.stats, scipy.fftpack

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
NumProcessors = comm.Get_size()


class KernelMachine(object):
    def __init__(self, kernel, 
                 lossfunction='squared',
                 regularizer='l2',
                 regparam=0,
                 randomfeatures=1000,
                 numfeaturepartitions=5,
                 TOL=1e-3,
                 MAXITER=100,
                 rho=1.0,
                 problem='multiclass_classification',
                 zerobased=False,
                 coefficients=None,
                 RFTs=None,         # Only for serializatoin
                 subtype='fast'):

        self.lossfunction = lossfunction
        self.regularizer = regularizer
        self.regparam = regparam
        self.randomfeatures = randomfeatures
        self.kernel = kernel
        self.numfeaturepartitions = numfeaturepartitions
        self.TOL = TOL
        self.MAXITER = MAXITER
        self.rho = rho
        self.coefficients = coefficients
        self.problem = problem
        self.subtype = subtype
        self.zerobased = zerobased
        self.RFTs = RFTs

    def save(self, outputfile):
        f = open(outputfile,'wb')
        cPickle.dump(self.__dict__,f)
        #cPickle.dump(self, f)
        f.close()

    def load(self, adict):
        self.__dict__ = adict

    def predict(self, X): #X is local numpy matrix
        N = int(self.numfeaturepartitions)
        W = self.coefficients
        D,k = W.shape
        n = X.shape[0]
        o = numpy.zeros((n,k))
        blksize = math.ceil(D / N)

        results = []
        for j in range(0,N):
            start = int(j * blksize)
            finish = int(min((j + 1) * blksize, D))
            JJ = range(start, finish)
            Dj = len(JJ)
            Z = (self.RFTs[j] / X) * math.sqrt(Dj / D)
            o = o + numpy.dot(Z, W[JJ,:])

        results.append(o)

        if self.problem=="multiclass_classification":
            pred = skylark.ml.utils.dummydecode(numpy.array(o), self.zerobased)
            results.append(pred)

        if self.lossfunction=="crossentropy":
            # implement probabilities
            pass

        return tuple(results)

    def train(self, data):
        (X,Y) = data
        lossfunction = loss(self.lossfunction)
        regularization = regularizer(self.regularizer)
        prox_loss = proxoperator(self.lossfunction)
        prox_regularizer = proxoperator(self.regularizer)


        # dimensions of the problem
        d = X.Width
        n = X.Height
        if self.problem=="multiclass_classification":
            k = int(comm.allreduce(max(Y.Matrix), op=MPI.MAX)) # number of classes
            if self.zerobased:
                k = k + 1
        else:
            k = Y.Matrix.shape[1]

        N = int(self.numfeaturepartitions) # number of column splits
        P = NumProcessors
        D = self.randomfeatures

        if rank==0:
            print self.__dict__
            print """Dimensions: X is n=%d x d=%d, k=%d classes, D=%d random features, P=%d processors, N=%d feature partitions""" % (n,d,k,D,P,N)
            starttime = MPI.Wtime()

        # Prepare ADMM intermediate matrices

        # distributed intermediate matrices -> split over examples
        O = elem.DistMatrix_d_VC_STAR()
        elem.Zeros(O, n, k)
        Obar = elem.DistMatrix_d_VC_STAR()
        elem.Zeros(Obar, n, k)
        nu = elem.DistMatrix_d_VC_STAR()
        elem.Zeros(nu, n, k)

        # distributed intermediate matrices -> split over features
        #W = elem.DistMatrix_d_VC_STAR();
        #elem.Zeros(W, D,k)
        #Wbar = elem.DistMatrix_d_STAR_STAR(); # (*,*) distribution - replicated everywhere
        #elem.Zeros(Wbar, D, k)
        #mu = elem.DistMatrix_d_VC_STAR();
        #elem.Zeros(mu, D, k)
        #J = range(W.ColShift, D, W.ColStride) # the rows of W,Wbar,mu owner local

        # on root node
        if rank==0:
            W = numpy.zeros((D,k));
            Wbar = numpy.zeros((D,k));
            mu = numpy.zeros((D,k));
        else:
            W = None
            Wbar = None
            mu = None

        # local intermediate matrices
        Wi = numpy.zeros((D,k));
        mu_ij = numpy.zeros((D,k));
        ZtObar_ij = numpy.zeros((D,k));

        iter = 0
        ni = O.LocalHeight


        # Create RFTs
        blksize = int(math.ceil(D / N))
        self.RFTs = [self.kernel.rft(blksize, self.subtype) for i in range(N-1)]
        self.RFTs.append(self.kernel.rft(D - (N - 1) * blksize, self.subtype))

        Precomputed = []

        #y = preprocess_labels(Y.Matrix)
        if self.lossfunction=="crossentropy" or self.lossfunction=="hinge":
            if not self.zerobased:
                y = Y.Matrix - 1.0 # convert from 1-to-K to 0-to-(K-1) representation
        else:
            y = skylark.ml.utils.dummycoding(Y.Matrix, k, self.zerobased)
            y = 2*y-1

        localloss = lossfunction(O.Matrix, y)

        while(iter < self.MAXITER):
            iter = iter + 1;

            totalloss = comm.reduce(localloss)
            if rank==0:
                ElapsedTime = MPI.Wtime() - starttime
                print 'iter=%d, objective=%f, time=%f' % (iter, totalloss + self.regparam*regularization(W), ElapsedTime);
                #print '\t\titer=%d, objective=%f' % (iter, objective(Wbar));

            Wbar=comm.bcast(Wbar, root=0)
            mu_ij = mu_ij - Wbar
            #mu_ij = mu_ij - Wbar.Matrix;

            # O optimization
            O.Matrix[:] = prox_loss(Obar.Matrix - nu.Matrix, 1.0/self.rho, y, O.Matrix[:]);

            # Compute value of Loss function

            # W optimization

            #W.Matrix[:] = prox_regularizer(Wbar.Matrix[J,:] - mu.Matrix, self.regparam/self.rho);
            if rank==0:
                W = prox_regularizer(Wbar - mu, self.regparam/self.rho);


            # graph projection step
            sum_o = numpy.zeros((ni, k));


            for j in range(0, N):
                start = j * blksize
                finish = min((j + 1) * blksize, D)
                JJ = range(start, finish)
                Dj = len(JJ)

                Z = (self.RFTs[j] / X.Matrix) * math.sqrt(Dj / D)
                if iter==1:
                    ZtZ = numpy.dot(Z.T, Z)
                    A = linalg.inv(ZtZ + numpy.identity(Dj))
                    Precomputed.append(A)

                ##############3 graph projection ##############
                #(Wi[JJ,:], o) = proj_graph(TransformOperator, X.Matrix, JJ, Wbar.Matrix[JJ, :] -  mu_ij[JJ,:],  ZtObar_ij[JJ,:] + Z(I,JJ)'*nu.Matrix, Precomputed[j]);
                C = Wbar[JJ, :] -  mu_ij[JJ,:]
                ZtD = ZtObar_ij[JJ,:] + numpy.dot(Z.T, nu.Matrix)

                WW = numpy.dot(Precomputed[j], (C+ZtD));
                Wi[JJ,:] = WW
                o = numpy.dot(Z, WW);

                ###############################################

                mu_ij[JJ,:] = mu_ij[JJ,:] + Wi[JJ,:];

                ZtObar_ij[JJ,:] = numpy.dot(Z.T, o);

                sum_o = sum_o + o;

            localloss = 0.0
            o = numpy.zeros((ni, k));
            for j in range(0,N):
                start = j * blksize
                finish = min((j + 1) * blksize, D)
                JJ = range(start, finish)
                Dj = len(JJ)
                Z = (self.RFTs[j] / X.Matrix) * math.sqrt(Dj / D)
                ZtObar_ij[JJ,:] = ZtObar_ij[JJ,:] + numpy.dot(Z.T, (O.Matrix - sum_o))/(N+1);
                o = o + numpy.dot(Z, Wbar[JJ,:])
            localloss = localloss + lossfunction(o, y)

            Obar.Matrix[:] = (sum_o + N*O.Matrix)/(N+1);
            nu.Matrix[:] = nu.Matrix + O.Matrix - Obar.Matrix;

            Wisum = comm.reduce(Wi)
            if rank==0:
                #Wisum = comm.allreduce(Wi)
                #Wbar.Matrix[J,:] = (Wisum[J,:] + W.Matrix)/(P+1)
                #Wbar.Matrix = (Wisum[J,:] + W.Matrix)/(P+1)
                Wbar = (Wisum + W)/(P+1)
                mu = mu + W - Wbar;
                # distributed sum below
                #mu.Matrix[:] = mu.Matrix + W.Matrix - Wbar.Matrix[J,:];

            comm.barrier()


        self.coefficients = Wbar
