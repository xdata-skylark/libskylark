#!/usr/bin/env python
# Usage: python example_krr_polynomial.py datasets/usps.t 1e-6 3 0 500 1500

import skylark, skylark.io
from skylark.ml import kernels
from skylark.ml.nonlinear import *
import pkg_resources, sys
import math

filename = pkg_resources.resource_filename("skylark", sys.argv[1]) 
X, Y = skylark.io.libsvm(filename).read()
regularization= float(sys.argv[2])
q = int(sys.argv[3])
c = float(sys.argv[4])
randomfeatures = int(sys.argv[5])
trn = int(sys.argv[6])

# For now we support only dense matrices in pure python
X = X.todense()
X = X / math.sqrt(X.shape[1]) # some nomralization to avoid things blowing up.

kernel = kernels.Polynomial(X.shape[1], q, c)
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

model = sketchpcr(kernel)
model.train(X[1:trn,:], Y[1:trn], randomfeatures/2)    
predictions = model.predict(X[trn+1:,:])
accuracy = skylark.metrics.classification_accuracy(predictions, Y[trn+1:])
print "SketchedPCR Accuracy=%f%%" % accuracy
    
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
    
