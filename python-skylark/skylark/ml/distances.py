import skylark.errors
import numpy
import scipy.sparse

def _multiply(X, Y):
  """
  Element-wise multipication of X and Y that works with X and Y either numpy
  or scipy csr/csc sparse matrix.
  """
  if type(X) != type(Y):
    raise skylark.errors.ParameterMistmatchError("X and Y must have same type")
  
  if X.shape != Y.shape:
    raise skylark.errors.ParameterMistmatchError("X and Y must have same shape")

  if isinstance(X, numpy.ndarray):
    return numpy.multiply(X, Y)

  if scipy.sparse.issparse(X):
    return X.multiply(Y)

  raise skylark.errors.UnsupportedError("Matrix type not supported")

def euclidean(X, Y):
  """
  euclidean(X, Y)
  
  Create a euclidean distance matrix (actually returns distance squared)
  
  Parameters
  ----------
  
  X: m x n matrix (sparse or dense)
  Y: t x n matrix (sparse or dense)
  
  Returns
  ---------
  
  D: t x m distance matrix D[i,j] is the squared distance between Y[i,:] and X[j,:]
  
  """
  m, n = X.shape
  norms_X = _multiply(X, X).dot(numpy.ones((n, 1)))
  ones_X  = numpy.ones((m, 1))
        
  t, n = Y.shape  
  norms_Y = _multiply(Y, Y).dot(numpy.ones((n, 1)))
  ones_Y = numpy.ones((t, 1))
  
  D = numpy.dot(ones_Y, norms_X.T) + numpy.dot(norms_Y, ones_X.T) - 2 * Y.dot(X.T)
    
  return D
