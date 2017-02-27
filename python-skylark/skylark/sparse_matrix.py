import lib
from ctypes import byref, c_void_p


class SparseMatrix(object):
  """  This implements a very crude CSC sparse matrix container only intended to
    hold local sparse matrices.
  
    Row indices are not sorted.
    Structure is always constants, and can only be attached by Attached.
    Values of non-zeros can be modified.
  """

  def __init__(self, size=0):
    self._obj = c_void_p()
    lib.callsl("sl_create_sparse_matrix", "SparseMatrix", size, byref(self._obj))    
  