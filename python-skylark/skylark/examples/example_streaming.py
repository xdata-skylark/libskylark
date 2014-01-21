#!/usr/bin/env python

import skylark.streaming, skylark.io, skylark.nla, skylark.ml.linear, skylark.ml.utils
import scipy.sparse.linalg
import numpy
import sys
 
input_filename = sys.argv[1]
sketchsize = int(sys.argv[2])
ncolumns = int(sys.argv[3])
mode = int(sys.argv[4]) # -K for regression problems with K outputs, 2 for binary classification and K>2 for multiclass classification  
regparam = float(sys.argv[5])
test_filename = sys.argv[6]
out_file = sys.argv[7]

#### In progress: simply call out to LSQR block by block for multiclass problems
#### 

def train(X,Y, regparam, mode=0, tolerance=1e-14, blocksize = 10):
        (m,n) = X.shape
        def matmat(V):
            P = X*V
            Q = numpy.sqrt(regparam)*V
            return numpy.concatenate((P, Q), axis = 0)
        def rmatvec(V):
            A = V[0:m,:]
            P = X.T*A;
            Q = numpy.sqrt(regparam)*V[m:,];
            return P + Q 
            
            #return numpy.concatenate((P, Q), axis = 1)
         
        Xoperator = scipy.sparse.linalg.LinearOperator((m+n,n), matvec=None, matmat=matmat, rmatvec=rmatvec)
         
        if mode > 2: # multiclass classification
            print "Mode 2"
            print Y.shape
            k = int(numpy.amax(Y))
            
            blocks = range(0,k,blocksize)
            blocks[-1] = k
            model = numpy.zeros((n, k))
            print "k=", k, blocks
            for i in range(0, len(blocks)-1):
                currentblock = range(blocks[i], blocks[i+1])
                print "Running over current block of classes:", currentblock
                k1 = len(currentblock)
                y = -1.0*numpy.ones((m, k1))
                t = 0
                for i in currentblock:
                    I = numpy.argwhere(Y==i+1)
                    y[I,t] = 1.0
                    t = t + 1
                
                y2 = numpy.concatenate((y, numpy.zeros((n, k1))), axis=0)
                print y2.shape
                model[:,currentblock], flag, iterations = skylark.nla.lsqr(Xoperator, y2, X=None, tol = tolerance)
            
        else:
            m, k = Y.shape
            Y = numpy.concatenate((Y, numpy.zeros((n, k))), axis=0)
            model, flag, iterations = skylark.nla.lsqr(Xoperator, Y, X=None, tol = tolerance)
        
        return model

def predict(Xt, W, mode = 0):
        pred = Xt*W
        labels = pred
        if mode == 2:
            labels = numpy.sign(pred)
        if mode > 2:
            labels = numpy.argmax(pred, axis = 1)
        return pred, labels

def evaluate(labels, Y, mode = 0):
        if mode <= 0:
            accuracy = numpy.linalg.norm(labels - Y)/numpy.linalg.norm(Y) 
        else: 
            accuracy = sum(labels==Y)*100.0/len(Y)
        return accuracy

def stream_evaluate(model, DataIterator, mode = 0):
    correct = 0.0
    total = 0.0
    total_test = 0
    for (Xt, Yt) in DataIterator:   
        m, n = Xt.shape
        predictions, labels = predict(Xt, model, mode)
        if mode <= 0:
            correct = correct + numpy.linalg.norm(labels - Yt)**2
            total = total + numpy.linalg.norm(Yt)**2
            accuracy = numpy.sqrt(correct)/numpy.sqrt(total)
            print "Test Relative error: %f / %f = %f " % (correct, total, accuracy)
            total_test = total_test + m
            print "Mean square error = %f" % (correct*1.0/total_test)
            
        else:
            correct = correct + numpy.sum(labels==(Yt.squeeze()-1))
            total = total + m
            accuracy = correct*100.0/total
            print "Test Accuracy: %d / %d = %f" % (correct, total, accuracy)
    return accuracy 


if sketchsize > 0:
    S = skylark.streaming.CWT(sketchsize)
    DataIterator = skylark.io.streamlibsvm2scipy(input_filename, ncolumns)
    (SX, SY) = S.sketch(DataIterator, ncolumns, mode)
    print >>sys.stdout, "Training the model..."
    sys.stdout.flush()
    model = train(SX, SY, regparam, -mode)
    print >>sys.stdout, "Training done."

else:
    for (SX, SY) in skylark.io.streamlibsvm2scipy(input_filename, ncolumns, blocksize=numpy.infty):
        pass
    print >>sys.stdout, "Training the model..."
    sys.stdout.flush() 
    model = train(SX, SY, regparam, mode)
    print >>sys.stdout, "Training done."
 
print >>sys.stdout, "Starting Testing..10000 at a time"
TestDataIterator = skylark.io.streamlibsvm2scipy(test_filename, ncolumns, blocksize=10000)

accuracy = stream_evaluate(model, TestDataIterator, mode)
    
numpy.savetxt(out_file, model)
