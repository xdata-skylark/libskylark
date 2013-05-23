'''
Created on Oct 26, 2012

@author: vikas
'''
import numpy 
import skylark
from distances import euclidean
import sys

def gaussian(X, Xtest=None,sigma=1.0):
        """
        gaussian(X, Xtest=None,sigma=1.0)
        
        Returns dense Gram matrix for Gaussian Kernels evaluated over datapoints

        Parameters:
        -------
        X  : m-by-n data matrix where m is number of examples, n is number of features
        Xt : Optional t x n matrix where t is number of test examples
        sigma : bandwidth of the Gaussian kernel

        Returns:
        -------
        m x m Gram matrix over X (if Xt is not provided)
        t x m Gram matrix between Xt and X if X is provided
        """


        if Xtest is None:
                K = numpy.exp(-euclidean(X,X)/(2*sigma**2))
        else:
                K = numpy.exp(-euclidean(X, Xtest)/(2*sigma**2))

        return K


        