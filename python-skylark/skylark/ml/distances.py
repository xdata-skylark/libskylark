'''
Created on Oct 26, 2012

@author: vikas
'''
import numpy

def euclidean(X,Y):
    """
    euclidean(X,Y)
    
        Create a euclidean distance matrix (actually returns distance squared)
    
    Parameters
    ----------
    
    X: m x n matrix
    Y: t x n matrix
    
    Returns
    ---------
    
    D: t x m distance matrix D[i,j] is the squared distance between Y[i,] and X[j,]
    
    """
    m , n = X.shape
    
    norms = X.multiply(X)*numpy.ones((n,1))
    ones  = numpy.ones((m, 1))
        
    t , n = Y.shape
    
    norms_t = Y.multiply(Y)*numpy.ones((n,1))
    ones_t = numpy.ones((t, 1))
    
    #D = numpy.dot(ones_t, norms.T) + numpy.dot(norms_t, ones.T) - 2*numpy.dot(Y, X.T)
    D = numpy.dot(ones_t, norms.T) + numpy.dot(norms_t, ones.T) - 2*Y*X.T
    
    return D