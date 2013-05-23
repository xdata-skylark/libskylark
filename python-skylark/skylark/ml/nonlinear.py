import numpy, numpy.random, scipy.linalg, scipy.stats
import kernels
import utils
from math import sqrt, cos, pi
import skylark.io, skylark.metrics, skylark.sketch
import sys

class rls(object):
	"""
	Class for solving Non-linear Regularized Least Squares problems using Gaussian Kernels.
	
	Examples
	---------
	
	Read a digit classification dataset
	
	>>> X,Y = skylark.read.sparselibsvm('')
	
	Set Regularization Parameter and Gaussian Kernel Bandwidth
	
	>>> regularization = 0.001
	>>> bandwidth = 10.0
	
	Build a model on 1000 training examples
	
	>>> model = rls()
	>>> model.train(X[1:1000,:],Y[1:1000],regularization,bandwidth)	
	
	Make Predictions on 1000 test examples
	
	>>> predictions = model.predict(X[1001:2000,:])
	
	Compute classification accuracy
	
	>>> accuracy = skylark.metrics.classification_accuracy(predictions, Y[1001:2000])
	>>> print "RLS Accuracy=%f%%" % accuracy
	RLS Accuracy=92.792793%
	
	"""
	def __init__(self):
		self.model = {}
		
	def train(self,X,Y, regularization=1, bandwidth=1, multiclass=True):
		"""
		Train a Regularized Least Squares model with Gaussian Kernel
		
		train(self,X,Y, regularization=1, bandwidth=1, multiclass=True)
		
		Parameters
        ----------
        X: m x n input matrix
        
        Y: m x 1 label vector (if multi-class classification problem, labels are from 0 to K-1 - trains one-vs-rest)
        
        regularization: regularization parameter 
        
        bandwidth: Gaussian kernel bandwidth i.e., K(x,z) = exp(-||x-z||^2/(2bandwidth^2))
        
        Returns
        --------
        Nothing. Internally sets the model parameters.
        
		"""
		m,n = X.shape
		K = kernels.gaussian(X,None,sigma=bandwidth)
		I = numpy.identity(m)
		if multiclass:
			Y = utils.dummycoding(Y).todense()
			Y = 2*Y - 1
		A = K + regularization*I
		alpha = scipy.linalg.solve(A, Y, sym_pos=True)
		self.model = {"bandwidth": bandwidth, "alpha": alpha, "regularization": regularization, "data": X, "multiclass":multiclass}
	
	def predict(self, Xt):
		"""
		Make Predictions on test data
		
		predict(Xt)
		
		Parameters
        ----------
        Xt: m x n input test matrix
        
        Returns
        -------
        m x 1 array of predictions on the test set.
		"""
		K = kernels.gaussian(self.model["data"], Xt, self.model["bandwidth"])
		pred = K*self.model["alpha"]
		if self.model["multiclass"]:
			pred = numpy.argmax(numpy.array(pred), axis=1)+1
		return pred

class sketchrls(object):
	"""
	Class for sketching based Non-linear Regularized Least Squares problems using Gaussian Kernels.
	
	The approach is based on [7]_
	
	Generate Randomized Feature maps associated with the Gaussian Kernel
            
        
        References
        -------------
        
        .. [7] A. Rahimi and B. Recht, Random Features for Large-scale Kernel Machines, NIPS 2009
	
	Examples
	---------
	
	Read a digit classification dataset
	
	>>> X,Y = skylark.read.sparselibsvm('')
	
	Set Regularization Parameter and Gaussian Kernel Bandwidth
	
	>>> regularization = 0.001
	>>> bandwidth = 10.0
	
	Set number of random features
	
	>>> random_features = 100
	
	Build a model on 1000 training examples
	
	>>> model = sketchrls()
	>>> model.train(X[1:1000,:],Y[1:1000],regularization, bandwidth, random_features)	
	
	Make Predictions on 1000 test examples
	
	>>> predictions = model.predict(X[1001:2000,:])
	
	Compute classification accuracy
	
	>>> accuracy = skylark.metrics.classification_accuracy(predictions, Y[1001:2000])
	>>> print "SketchedRLS Accuracy=%f%%" % accuracy
	SketchedRLS Accuracy=86.386386%
	
	"""
	
	def __init__(self, seed=123):
		self.model = {}
		self.seed = 123
		
		
	def sketch(self, A):
		"""
		Implements the Rahimi-Recht sketch on a given matrix A
		
		"""
		
		m,n = A.shape
		k = self.random_features
		Z = self.mysketch.sketch(A, k, 'right')
		Z = Z*(sqrt(k)/self.bandwidth)
		b = self.bias
		ones_m = numpy.matrix(numpy.ones((m,1)))
		Z = sqrt(2.0/k) * numpy.cos(Z + ones_m*b.T) # note: numpy.cos works elementwise unlike math.cos
		return Z

		
	def train(self,X,Y, regularization=1, bandwidth=1, random_features=100, multiclass=True):
		"""
		Train an RLS model with sketching primitives
		
		Parameters
        ----------
        X: m x n input matrix
        
        Y: m x 1 label vector (if multi-class classification problem, labels are from 0 to K-1 - trains one-vs-rest)
 		
 		regularization: regularization parameter 
        
        bandwidth: Gaussian kernel bandwidth i.e., K(x,z) = exp(-||x-z||^2/(2bandwidth^2))
        
        randomfeatures: how many random fourier features to generate
        
        Returns
        --------
        Nothing. Internally sets the model parameters.
        
		"""
		m,n = X.shape
		self.random_features = random_features
		self.bandwidth = bandwidth
		self.mysketch = skylark.sketch.JL(self.seed)
		self.bias = numpy.matrix(numpy.random.uniform(0, 2*pi, (random_features,1)))
		Z = self.sketch(X)
		
		I = numpy.identity(random_features)
		if multiclass:
			Y= utils.dummycoding(Y).todense()
			Y = 2*Y - 1
			
		A = Z.T*Z + regularization*I
		weights = scipy.linalg.solve(A, Z.T*Y, sym_pos=True)
		self.model = {"bandwidth": bandwidth, "weights": weights, "random_features": random_features,  
					"regularization": regularization, "multiclass":multiclass}
	
	def predict(self, Xt):
		"""
		Make predictions on test data
		
		Parameters
        ----------
        Xt: m x n input test matrix
        
        Returns
        -------
        m x 1 array of predictions on the test set.
		"""
		Zt = self.sketch(Xt)
		pred = Zt*self.model["weights"]
		if self.model["multiclass"]:
			pred = numpy.argmax(numpy.array(pred), axis=1)+1
		return pred

class nystromrls(object):
	
	def __init__(self, seed=123):
		self.model = {}
		self.seed = 123
		
	def train(self,X,Y, regularization=1, bandwidth=1, random_features=100, probdist = 'uniform', multiclass=True):
		"""
		type = 'uniform' | 'leverages' | 'fourier_leverages'
		l: number of Nystrom random samples to take
		k: rank-k approximation to the Gram matrix of the sampled data is used
		"""
		m,n = X.shape
		
		nz_values = range(0, m)
		
		#uniform
		if probdist == 'uniform':
			nz_prob_dist = numpy.ones((m,1))/m
		if probdist ==  'leverages':
			K = kernels.gaussian(X,None,sigma=bandwidth)
			Im = numpy.identity(m)
			nz_prob_dist = numpy.diag(K*scipy.linalg.inv(K+regularization*Im))
			nz_prob_dist = nz_prob_dist/sum(nz_prob_dist)
		indices = scipy.stats.rv_discrete(values=(nz_values, nz_prob_dist), name = 'uniform').rvs(size=random_features)
		K_II = kernels.gaussian(X[indices, :], None, sigma = bandwidth)
		I = numpy.identity(random_features)
		eps = 1e-8
		(evals, evecs) = scipy.linalg.eigh(K_II + eps*I)
		Xtrain = X[indices, :]
		Z  =  kernels.gaussian(Xtrain, X, sigma = bandwidth)
		U = (evecs*numpy.diagflat(1.0/numpy.sqrt(evals)))
		Z = Z*U
		if multiclass:
			Y= utils.dummycoding(Y).todense()
			Y = 2*Y - 1
			
		A = Z.T*Z + regularization*I
		weights = scipy.linalg.solve(A, Z.T*Y, sym_pos=True)
		self.model = {"bandwidth": bandwidth, "weights": weights, "random_features": random_features,  
					"regularization": regularization, "multiclass":multiclass, "Xtrain":Xtrain, "U":U }
	
	def predict(self, Xt):
		"""
		Make predictions on test data
		
		Parameters
        ----------
        Xt: m x n input test matrix
        
        Returns
        -------
        m x 1 array of predictions on the test set.
		"""
		Zt =  kernels.gaussian(self.model["Xtrain"], Xt, sigma = self.model["bandwidth"])*self.model["U"]
		pred = Zt*self.model["weights"]
		if self.model["multiclass"]:
			pred = numpy.argmax(numpy.array(pred), axis=1)+1
		return pred


if __name__=="__main__":
	X,Y = skylark.io.sparselibsvm(sys.argv[1])
	regularization= float(sys.argv[2])
	bandwidth = float(sys.argv[3])
	randomfeatures = int(sys.argv[4])
	trn = int(sys.argv[5])
	
	model = rls()
	print len(Y[1:trn]), len(Y[trn+1:])
	
	model.train(X[1:trn,:],Y[1:trn],regularization,bandwidth)	
	predictions = model.predict(X[trn+1:,:])
	accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
	print "RLS Accuracy=%f%%" % accuracy
	
	model = sketchrls()
	
	model.train(X[1:trn,:],Y[1:trn],regularization,bandwidth,randomfeatures)	
	predictions = model.predict(X[trn+1:,:])
	accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
	print "SketchedRLS Accuracy=%f%%" % accuracy
	
	model = nystromrls()
	model.train(X[1:trn,:],Y[1:trn],regularization,bandwidth,randomfeatures, probdist = 'uniform')
	predictions = model.predict(X[trn+1:,:])
	accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
	print "Nystrom uniform Accuracy=%f%%" % accuracy
	
	
	model = nystromrls()
	model.train(X[1:trn,:],Y[1:trn],regularization,bandwidth,randomfeatures, probdist = 'leverages')
	predictions = model.predict(X[trn+1:,:])
	accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
	print "Nystrom leverages Accuracy=%f%%" % accuracy
	
	
	