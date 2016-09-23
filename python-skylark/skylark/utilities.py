'''
Created on Dec 7, 2012

@author: vikas
'''

import numpy

def norms(A, dimension=0):
    """
    Parameters
    ------------
    A : m x n matrix
    dimension : 0 returns euclidean norms of rows and 1 returns euclidean norms of columns
    """
    
    rownorms = numpy.apply_along_axis(numpy.linalg.norm, 1, A)
    
    return rownorms