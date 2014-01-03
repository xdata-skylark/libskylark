#!/usr/bin/env python
# Usage: python example_krr.py datasets/usps.t 0.001 10 500 1500

import skylark, skylark.io
from skylark.ml.nonlinear import *
import pkg_resources, sys

filename = pkg_resources.resource_filename("skylark", sys.argv[1])    
X,Y = skylark.io.sparselibsvm2scipy(filename)
regularization= float(sys.argv[2])
bandwidth = float(sys.argv[3])
randomfeatures = int(sys.argv[4])
trn = int(sys.argv[5])
    
model = rls()  
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
    
