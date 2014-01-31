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

class Fastfood(object):
	def __init__(self, dimensions, kernel='gaussian', kernelparam=1.0, randomfeatures=1000, SEED=45678):
                self.kernel = kernel
                self.kernelparam = kernelparam
                self.randomfeatures = randomfeatures
                self.SEED = SEED
                self.dimensions = dimensions
                numpy.random.seed(SEED)
                # currently only Gaussian
                #self.W = numpy.array(numpy.random.randn(self.dimensions, self.randomfeatures), order='F');
		self.blocks = int(math.ceil(self.randomfeatures*1.0/self.dimensions))
		self.finalrandomfeatures = self.blocks*self.dimensions		
		self.scale = 1.0/(kernelparam*math.sqrt(self.dimensions))


		self.B=numpy.zeros((self.blocks, self.dimensions))
		self.G=numpy.zeros((self.blocks, self.dimensions))
		self.P=numpy.zeros((self.blocks, self.dimensions), dtype=numpy.int32)
		binary = scipy.stats.bernoulli(0.5)
		for i in range(0,self.blocks):
			self.B[i,:] = 2.0*binary.rvs(self.dimensions) - 1.0
			self.G[i,:] = numpy.random.randn(1,self.dimensions)
			self.P[i,:] = numpy.random.permutation(self.dimensions)
		
	
		self.b = numpy.random.uniform(0, 2*math.pi, (1,self.finalrandomfeatures))


#	@profile
	def map(self, X, J=-1):
		X = numpy.asarray(X)
		m,n = X.shape

		if J==-1:
                        J = range(0,self.finalrandomfeatures)
		
		startblock = int(numpy.floor(J[0]*1.0/n))
		endblock = int(numpy.floor(J[-1]*1.0/n))

		Ztmp = numpy.zeros((m,n), order='F')
		Z1 = numpy.zeros((m, (endblock-startblock+1)*n), order='F')
		 
		for i in range(startblock, endblock+1):
			# row vector B times X uses numpy broadcast rules below; dct works rowwise
			# check if column assignment in Ztmp will causing the code to be slowish
			Ztmp[:,self.P[i,:]] = scipy.fftpack.dct(self.B[i,:]*X, norm='ortho')*math.sqrt(n);
			Z1[:,((i-startblock)*n):((i+1-startblock)*n)] = self.scale*scipy.fftpack.dct(self.G[i,:]*Ztmp, norm='ortho')*math.sqrt(n);

		Z = Z1[:, (J[0]-startblock*n):(J[-1]-startblock*n+1)]

		ones = numpy.ones((m,1));
	
		Z = Z + self.b[:,J] # using numpy broadcast rules
                Z = numpy.cos(Z)*math.sqrt(2.0/self.finalrandomfeatures);

		return Z	

 
# can be replaced by more efficient Fastfood like operations
class ExplicitFeatureMap(object):
	def __init__(self, dimensions, kernel='gaussian', kernelparam=1.0, randomfeatures=1000, SEED=1234):
		self.kernel = kernel
		self.kernelparam = kernelparam
		self.randomfeatures = randomfeatures
		self.SEED = SEED
		self.dimensions = dimensions
		numpy.random.seed(SEED)
		# currently only Gaussian
		self.W = numpy.array(numpy.random.randn(self.dimensions, self.randomfeatures), order='F');
	   	self.b = numpy.random.uniform(0, 2*math.pi, (1,self.randomfeatures))
#	@profile
	def map(self, X, J=-1):
		n,d = X.shape
		if J==-1:
			J = range(0,self.randomfeatures)
		ones = numpy.ones((n,1));
		w = self.W[:,J]
		Z2 = numpy.dot(X,w)/self.kernelparam;
		Z2 = Z2 + self.b[:,J] # using numpy broadcast rules 
		Z = numpy.cos(Z2)*math.sqrt(2.0/self.randomfeatures);
		return Z

	def covariance(self, X, J):
		Z = map(X,J)
		ZtZ = numpy.dot(Z.T, Z)
		return ZtZ

	def matmul(self, X, J, Y):
		Z = map(X,J)
		ZY = numpy.dot(Z,Y)
		return ZY

	def matmul_transp(self, X, J, Y):
		Z = map(X,J)
		ZtY = numpy.dot(Z.T, Y)
		return ZtY

 
class KernelMachine(object):
	def __init__(self, lossfunction='squared',
                      regularizer='l2',
                      regparam=0,
                      randomfeatures=1000,
                      kernel='gaussian',
                      kernelparam=1.0,
                      numfeaturepartitions=5,
                      TOL=1e-3,
                      MAXITER=100,
                      SEED=12345,
		      rho=1.0,
		      problem='multiclass_classification',
		      coefficients=None,
		      TransformOperator=None, fastfood=True):

		self.lossfunction = lossfunction
		self.regularizer = regularizer
		self.regparam = regparam
		self.randomfeatures = randomfeatures
		self.kernel = kernel
		self.kernelparam = kernelparam
		self.numfeaturepartitions = numfeaturepartitions
		self.TOL = TOL
		self.MAXITER = MAXITER
		self.SEED = SEED
		self.rho = rho
		self.coefficients = coefficients
		self.problem = problem
		self.TransformOperator = TransformOperator
		self.fastfood = fastfood

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

		results = []
		for j in range(0,N):
			start = int(math.floor(numpy.round(j*D*1.0/N)))
                        finish = int(math.floor(numpy.round((j+1)*D*1.0/N)))
                        JJ = range(start, finish)
                        Z = self.TransformOperator.map(X, JJ)
			o = o + numpy.dot(Z, W[JJ,:])

		results.append(o)

		if self.problem=="multiclass_classification":
			pred = numpy.argmax(numpy.array(o), axis=1)+1		
			results.append(pred)

		if self.lossfunction=="crossentropy":
			# implement probabilities
			pass

		return tuple(results)		

 #       @profile
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
			# Instiantiate an explict feature map, but compute with it implicitly
		if self.fastfood:
			self.TransformOperator = Fastfood(d, self.kernel, self.kernelparam, self.randomfeatures, self.SEED)
		else:
			self.TransformOperator = ExplicitFeatureMap(d, self.kernel, self.kernelparam, self.randomfeatures, self.SEED)
		
		Precomputed = []
			
		#y = preprocess_labels(Y.Matrix)
		if self.lossfunction=="crossentropy" or self.lossfunction=="hinge":
			y = Y.Matrix - 1.0 # convert from 1-to-K to 0-to-(K-1) representation
		else:
			y = skylark.ml.utils.dummycoding(Y.Matrix, k)
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
				start = int(math.floor(numpy.round(j*D*1.0/N)))
				finish = int(math.floor(numpy.round((j+1)*D*1.0/N))) - 1
	            JJ = range(start, finish)
				Dj = len(JJ)

				Z = self.TransformOperator.map(X.Matrix, JJ)
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
				start = int(math.floor(numpy.round(j*D*1.0/N)))
                                finish = int(math.floor(numpy.round((j+1)*D*1.0/N))) - 1
                                JJ = range(start, finish)
                                Dj = len(JJ)
				Z = self.TransformOperator.map(X.Matrix, JJ)
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
