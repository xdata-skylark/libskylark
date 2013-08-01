import numpy, scipy, scipy.sparse

def dummycoding(Y, K=-1):
	"""
	Y has discrete 1.to.K labels.
	Returns an indicator sparse matrix 
	K = -1 infers the number of classes from Y. K>0 uses K classes for the coding (asusming that in streaming or testing settings, not all classes may be represented in Y
	"""
	
	Y = numpy.array(Y-1, dtype=int)
	m = len(Y)
	if K == -1:
		n = max(Y)+1
	else:
		n = K
		
	data = numpy.ones(m)
	col = Y.squeeze()
	row = scipy.arange(m)

	Y = scipy.sparse.csr_matrix( (data, (row, col)), shape = (m,n))
	
	return Y.todense()

