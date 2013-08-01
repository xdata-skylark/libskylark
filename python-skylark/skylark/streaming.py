from skylark.sketch import SketchingTransform
import numpy.random
import scipy.sparse, scipy.stats

class CWT(object):
    def __init__(self, s, context=123):
        self.context = context
        self.sketchsize = s
        numpy.random.seed(self.context)
        
    def sketch(self, Aiterator, otherdimension, nclasses = 0):
        s = self.sketchsize
        SX = scipy.sparse.csr_matrix((s, otherdimension))
        if nclasses > 2:
            SY = numpy.zeros(s, nclasses)
        else:
            SY = numpy.zeros(s)
        nz_values=[-1,+1]
        nz_prob_dist=[0.5,0.5]
        
        for (X,Y) in Aiterator:
            (n, d) = X.shape
            D = scipy.stats.rv_discrete(values=(nz_values, nz_prob_dist), name = 'dist').rvs(size=n)
            H = scipy.stats.randint(0,s).rvs(n)
            X= X.tocoo()
            I = X.row
            J = X.col
            V = D[I]*(X.data)
            I = H[I]
            
            B = scipy.sparse.csr_matrix((V, (I,J)), shape=(s, otherdimension) )
            SX = SX + B
            
            S = scipy.sparse.csr_matrix( (D, (H, scipy.arange(n))), shape = (s,n))
            if nclasses > 2:
                SY = SY + S.dot(skylark.ml.utils.dummycode(Y, nclasses))
            else:
                SY = SY + S.dot(Y)
                
        return (SX, SY)
    
