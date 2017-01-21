import numpy
import skylark
import skylark.lib as lib
from ctypes import byref, c_void_p, c_double, c_int


# Model base
class Model(object):
    def __init__(self, direction, kernel):
        self._direction = direction
        self._kernel = kernel
        self._model = c_void_p()
    
    def load_matrix():
        pass

    def predict():
        pass
    
    def save(fname, header):
        pass
    

# TODO: Change name of the class... find a better name!
class Classifier(Model):
    pass
