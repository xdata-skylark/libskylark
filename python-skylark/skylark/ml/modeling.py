import skylark.sketch
import numpy, scipy
import json, math

class LinearizedKernelModel:
    """
    This class allows loading of model file created by skylark_ml into Python
    and making predicitions.
    """
    def __init__(self, fname):
        with open(fname, 'r') as f:
            s = ''
            for line in f:
                if line[0] != '#':
                    s = s + line
            f.close()

            data = json.loads(s)
            self._s = int(data['num_features'])
            self._d = int(data['input_size'])
            self._k = int(data['num_outputs'])
            self._W = numpy.fromstring(data['coef_matrix'], sep=' ').reshape((self._s, self._k))
            num_maps = int(data['feature_mapping']['number_maps'])
            self._maps = \
                [skylark.sketch.deserialize_sketch(data['feature_mapping']['maps'][str(i)]) for i in range(num_maps)]
            self._scale_maps = data['feature_mapping']['scale_maps'] == 'true'
            self._regression = data['regression'] == 'true'

    def get_input_dimension(self):
        return self._d

    def predict(self, X):
        n = X.shape[0]
        d = X.shape[1]
        D = numpy.zeros((n, self._k))
        end = 0
        if self._maps == []:
            D = X.dot(self._W)
        else:
            for S in self._maps:
                sj = S.getsketchdim()
                start = end
                end = start + sj
                
                Z = numpy.zeros((X.shape[0], sj))
                S.apply(X, Z, 'rowwise')
                Wv = self._W[start:end, :]
                if self._scale_maps:
                    D = D + math.sqrt(float(sj) / d) * scipy.dot(Z, Wv)
                else:
                    D = D + scipy.dot(Z, Wv)

        if self._regression:
            return D
        else:
            return D.argmax(axis = 1)
