#!/usr/bin/env python
# Usage: python example_krr.py datasets/usps.t 0.001 10 500 1500

import skylark, skylark.io
from skylark.ml import kernels
from skylark.ml.nonlinear import *
import pkg_resources, sys

filename = pkg_resources.resource_filename("skylark", sys.argv[1]) 
X, Y = skylark.io.libsvm(filename).read()
regularization= float(sys.argv[2])
bandwidth = float(sys.argv[3])
randomfeatures = int(sys.argv[4])
trn = int(sys.argv[5])

# For now we support only dense matrices in pure python
X = X.todense()
    
kernel = kernels.Gaussian(X.shape[1], bandwidth)
model = rls(kernel)  
model.train(X[1:trn,:], Y[1:trn],regularization)    
predictions = model.predict(X[trn+1:,:])
accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
print "RLS Accuracy=%f%%" % accuracy
    
model = sketchrls(kernel)
model.train(X[1:trn,:], Y[1:trn], randomfeatures, regularization)    
predictions = model.predict(X[trn+1:,:])
accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
print "SketchedRLS Accuracy=%f%%" % accuracy
    
model = nystromrls(kernel)
model.train(X[1:trn,:], Y[1:trn], randomfeatures, regularization, probdist = 'uniform')
predictions = model.predict(X[trn+1:,:])
accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
print "Nystrom uniform Accuracy=%f%%" % accuracy
        
model = nystromrls(kernel)
model.train(X[1:trn,:], Y[1:trn], randomfeatures, regularization, probdist = 'leverages')
predictions = model.predict(X[trn+1:,:])
accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
print "Nystrom leverages Accuracy=%f%%" % accuracy
    
