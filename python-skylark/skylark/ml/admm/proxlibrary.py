from _libproxoperators import *
import numpy
# Prox operators for Loss functions
squared = lambda O, a, T, Oinit:  (O + a*T)/(1.0+a) # not iterative, so doesnt need Oinit

def crossentropy(O,a,T, Oinit):
    epsilon = 1e-8
    MAXITER = 100
    flag = crossentropy_prox(T, O, a, Oinit, MAXITER, 1e-8, 0)
    return Oinit

def hinge(O,a,T, Oinit):
    epsilon = 1e-8
    MAXITER = 100
    flag = hinge_prox(T, O, a, Oinit)
    return Oinit

def lad(O,a,T,Oinit):
    X = T
    X[O>(T+a)] = O[O>(T+a)] - a
    X[O<(T-a)] = O[O<(T-a)] + a
    return X

# Prox operators for Regularizers
l2 = lambda T,a: T/(1.0+a)


prox_operators = {
        'squared': squared,
        'l2': l2,
        'crossentropy': crossentropy,
        'hinge': hinge,
        'lad': lad
        }

regularizers = {'l2': lambda W: 0.5*numpy.linalg.norm(W,'fro')**2 }
losses = {'squared': lambda O,Y: 0.5*numpy.linalg.norm(O-Y,'fro')**2,
          'crossentropy': lambda O,Y: crossentropy_obj(Y,O),
          'hinge': lambda O, Y: hinge_obj(Y,O),
          'lad': lambda O, Y: numpy.sum(numpy.absolute(O-Y))}

proxoperator = lambda function: prox_operators[function]

regularizer = lambda regularization : regularizers[regularization]
loss = lambda lossfunction: losses[lossfunction]
