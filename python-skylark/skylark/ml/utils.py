import numpy, scipy, scipy.sparse

def dummycoding(Y, K=None, zerobased=False):
  """
  Returns an indicator matrix that can be used for classification.

  :param Y: discrete input labels, 1.to.K (or 0.to.K-1 if zerobased is True)
  :param K: number of classes. Infers the number if None.
  :param zerobased: whether labels are zero based on 1 based.
  """

  Y = numpy.array(Y, dtype=int)
  if not zerobased:
    Y = Y - 1
  m = len(Y)
  if K is None:
    n = max(Y)+1
  else:
    n = K
		
  data = numpy.ones(m)
  col = Y.squeeze()
  row = scipy.arange(m)

  Y = scipy.sparse.csr_matrix( (data, (row, col)), shape = (m,n))
	
  return Y.todense()


def dummydecode(pred, zerobased=False):
  """
  Decode prediction on indicator matrix back to labels.

  :param Y: predicitons, number of classes is number of columns.
  :param zerobased: whether labels are zero based on 1 based.
  """
  pred = numpy.argmax(numpy.array(pred), axis=1)
  if not zerobased:
    pred = pred + 1
  return pred

