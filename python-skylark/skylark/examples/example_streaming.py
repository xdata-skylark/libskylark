import skylark.streaming, skylark.io
import scipy.sparse.linalg
import numpy
import sys
 
input_filename = sys.argv[1]
sketchsize = int(sys.argv[2])
ncolumns = int(sys.argv[3])
regparam = float(sys.argv[4])
test_filename = sys.argv[5]
out_file = sys.argv[6]

def train(X,Y, regparam):
        (m,n) = X.shape
        matvec = lambda v: X*v
        rmatvec = lambda v: X.T*v
        Xoperator = scipy.sparse.linalg.LinearOperator((m,n), matvec, rmatvec)
        w = scipy.sparse.linalg.lsqr(X, Y, damp = regparam, show = True)
        model = w[0]
        return model

def predict(Xt, w):
        pred = Xt*w
        return pred

def evaluate(pred, Y):
        accuracy = sum(numpy.sign(pred)==Y)*100.0/len(Y)
        return accuracy

if sketchsize > 0:
    S = skylark.streaming.CWT(sketchsize)
    DataIterator = skylark.io.streamlibsvm2scipy(input_filename, ncolumns)
    (SX, SY) = S.sketch(DataIterator, ncolumns)
else:
    (SX, SY) = skylark.io.sparselibsvm2scipy(input_filename)
 
model = train(SX, SY, regparam)

correct = 0
total = 0
for (Xt, Yt) in skylark.io.streamlibsvm2scipy(test_filename, ncolumns):   
    predictions = predict(Xt, model)
    correct = correct + sum(numpy.sign(predictions)==Yt)
    total = total + len(Yt)
    print "Test Accuracy: %d / %d = %f" % (correct, total, correct*100.0/total)
    

numpy.savetxt(out_file, model)
