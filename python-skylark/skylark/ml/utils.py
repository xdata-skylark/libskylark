import numpy, scipy, scipy.sparse

def dummycoding(Y):
	"""
	Y has discrete 1.to.K labels.
	Returns an indicator sparse matrix 
	"""
	Y = [int(y-1) for y in Y]
	m = len(Y)
	n = max(Y)+1
	data = numpy.ones(m)
	col = Y
	row = scipy.arange(m)
	Y = scipy.sparse.csr_matrix( (data, (row, col)), shape = (m,n))
	
	return Y
