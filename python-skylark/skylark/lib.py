import ctypes
from ctypes import byref, cdll, c_double, c_void_p, c_int, c_char_p, pointer, POINTER, c_bool
import ctypes.util

import errors
import numpy, scipy.sparse
import atexit
import time
import sys

_libc = cdll.LoadLibrary(ctypes.util.find_library('c'))
_libc.free.argtypes = (ctypes.c_void_p,)

# Function for initialization and reinitilialization
def initialize(seed=-1):
    """
    Reinitalize the library with a seed. If seed is -1 then system time is
    used to create the seed.
    """

    global lib, ctxt_obj, ELEM_INSTALLED, KDT_INSTALLED
    
    #
    # Load C-API library and set return types
    #
    lib = cdll.LoadLibrary('libcskylark.so')
    lib.sl_create_context.restype              = c_int
    lib.sl_create_default_context.restype      = c_int
    lib.sl_free_context.restype                = c_int
    lib.sl_create_sketch_transform.restype     = c_int
    lib.sl_serialize_sketch_transform.restype  = c_int
    lib.sl_deserialize_sketch_transform.restype = c_int
    lib.sl_wrap_raw_matrix.restype             = c_int
    lib.sl_free_raw_matrix_wrap.restype        = c_int
    lib.sl_wrap_raw_sp_matrix.restype          = c_int
    lib.sl_free_raw_sp_matrix_wrap.restype     = c_int
    lib.sl_raw_sp_matrix_nnz.restype           = c_int
    lib.sl_raw_sp_matrix_height.restype           = c_int
    lib.sl_raw_sp_matrix_width.restype           = c_int
    lib.sl_raw_sp_matrix_struct_updated.restype = c_int
    lib.sl_raw_sp_matrix_reset_update_flag.restype = c_int
    lib.sl_raw_sp_matrix_data.restype          = c_int
    lib.sl_strerror.restype                    = c_char_p
    lib.sl_supported_sketch_transforms.restype = c_char_p
    lib.sl_has_elemental.restype               = c_bool
    lib.sl_has_combblas.restype                = c_bool
    lib.sl_get_exception_info.restype          = None
    lib.sl_print_exception_trace               = None
    
    ELEM_INSTALLED = lib.sl_has_elemental()
    KDT_INSTALLED  = lib.sl_has_combblas()

    if seed == -1:
        seed = int(time.time())

    if 'ctxt_obj' in globals():
        lib.sl_free_context(ctxt_obj)
            
    ctxt_obj = c_void_p()
    lib.sl_create_default_context(seed, byref(ctxt_obj))


def finalize():
    """
    Finalize (de-allocate) the library. However, note that that will not cause
    allocated objects (e.g. sketch transforms) to be freed. They are freed by
    the garbage collector when detected as garbage (no references).
    """
    # TODO free dll (?)
    global lib, ctxt_obj
    if lib is not None:
        if ctxt_obj != 0:
            lib.sl_free_context(ctxt_obj)
        ctxt_obj = 0

# Make sure finalize is called before exiting (just in case).
atexit.register(finalize)

# Actually initialize the C-API.
initialize(int(time.time()))

def callsl(fname, *args):
    """
    Call the skylark function f (which is a string).
    """
    f = getattr(lib, fname)
    errno = f(*args)
    if errno != 0:
        c_errinfo = c_char_p()
        lib.sl_get_exception_info(byref(c_errinfo))
        errinfo = c_errinfo.value
        _libc.free(c_errinfo)
        raise errors.LowerLayerError(errinfo)

#
# Matrix type adapters: specifies how to interact with the underlying (perhaps in C/C++)
# data structure.
#
class NumpyAdapter:
  def __init__(self, A):
    if A.dtype.type is not numpy.float64:
      raise errors.UnsupportedError("Only float64 matrices are supported.")
    if A.base is not None:
      raise errors.UnsupportedError("Passing a numpy matrix view is not supported.")

    self._A = A

  def ctype(self):
    return "Matrix"

  def ptr(self):
    data = c_void_p()
    # If the matrix is kept in C ordering we are essentially wrapping the transposed
    # matrix
    if self.getorder() == "F":
      callsl("sl_wrap_raw_matrix", \
             self._A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
             self._A.shape[0], self._A.shape[1] if self._A.ndim > 1 else 1 , byref(data))
    else:
      callsl("sl_wrap_raw_matrix", \
             self._A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
             self._A.shape[1] if self._A.ndim > 1 else self._A.shape[0], \
             self._A.shape[0] if self._A.ndim > 1 else 1 , byref(data))
    self._ptr = data
    return data

  def ptrcleaner(self):
    callsl("sl_free_raw_matrix_wrap", self._ptr);

  def getdim(self, dim):
    return self._A.shape[dim]

  def getobj(self):
    return self._A

  def getorder(self):
    if self._A.flags.f_contiguous:
      return 'F'
    else:
      return 'C'

  def iscompatible(self, B):
    if isinstance(B, NumpyAdapter) and self.getorder() != B.getorder():
      return "sketching numpy array to numpy array requires same element ordering", None
    if isinstance(B, ScipyAdapter) and self.getorder() != B.getorder():
      return "sketching numpy array to scipy array requires same element ordering", None
    elif not isinstance(B, NumpyAdapter) and not isinstance(B, ScipyAdapter) and self.getorder() == 'C':
      return "numpy combined with non numpy/scipy must have fortran ordering", None
    else:
      return None, self._A.flags.c_contiguous

  def getctor(self):
    return NumpyAdapter.ctor

  @staticmethod
  def ctor(m, n, B):
    # Construct numpy array that is compatible with B. If B is a numpy or scipy array the
    # element order (Fortran or C) must match. For all others (e.g., Elemental
    # and KDT) the order must be Fortran because this is what the lower layers
    # expect.
    if isinstance(B, NumpyAdapter) or isinstance(B, ScipyAdapter):
      return numpy.empty((m,n), order=B.getorder())
    else:
      return numpy.empty((m,n), order='F')

class ScipyAdapter:
  def __init__(self, A):

    if isinstance(A, scipy.sparse.csr_matrix) or isinstance(A, scipy.sparse.csc_matrix):
        self._A = A
    else:
        self._A = scipy.sparse.csr_matrix(A)

  def ctype(self):
    return "SparseMatrix"

  def ptr(self):
    data = c_void_p()
    iptr = self._A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    cols = self._A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dptr = self._A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # If the matrix is kept in C ordering we are essentially wrapping the transposed
    # matrix
    if self.getorder() == "F":
      callsl("sl_wrap_raw_sp_matrix", \
             iptr, cols, dptr, len(self._A.indices), \
             self._A.shape[0], self._A.shape[1] if self._A.ndim > 1 else 1, byref(data))
    else:
      callsl("sl_wrap_raw_sp_matrix", \
             iptr, cols, dptr, len(self._A.indices), \
             self._A.shape[1] if self._A.ndim > 1 else self._A.shape[0], \
             self._A.shape[0] if self._A.ndim > 1 else 1 , \
             byref(data))
    callsl("sl_raw_sp_matrix_reset_update_flag", data)
    self._ptr = data
    return data

  def ptrcleaner(self):
    # before cleaning the pointer make sure to update the csr structure if
    # necessary.
    update_csc = c_bool()
    callsl("sl_raw_sp_matrix_struct_updated", self._ptr, \
           byref(update_csc))

    if(update_csc.value):
      # first we check the required size of the new structure
      nnz, m, n = (c_int(), c_int(), c_int())
      callsl("sl_raw_sp_matrix_nnz", self._ptr, byref(nnz))
      callsl("sl_raw_sp_matrix_height", self._ptr, byref(m))
      callsl("sl_raw_sp_matrix_width", self._ptr, byref(n))

      if isinstance(self._A, scipy.sparse.csc_matrix):
        self._A._shape = (m.value, n.value)
        indptrdim = self._A.shape[1] + 1 if self._A.ndim > 1 else 2
      else:
        self._A._shape = (n.value, m.value)
        indptrdim = self._A.shape[0] + 1 if self._A.ndim > 1 else 2

      indptr  = numpy.zeros(indptrdim, dtype='int32')
      indices = numpy.zeros(nnz.value, dtype='int32')
      values  = numpy.zeros(nnz.value)

      callsl("sl_raw_sp_matrix_data", self._ptr,
             indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), \
              indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), \
              values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

      self._A.__dict__["indptr"]  = indptr
      self._A.__dict__["indices"] = indices
      self._A.__dict__["data"]    = values

    callsl("sl_free_raw_sp_matrix_wrap", self._ptr);

  def getdim(self, dim):
    return self._A.shape[dim]

  def getobj(self):
    return self._A

  def getorder(self):
    if isinstance(self._A, scipy.sparse.csr_matrix):
      return 'C'     # C ordering -- CSR format
    else:
      return 'F'     # Fortran ordering - CSC format

  def iscompatible(self, B):
    if isinstance(B, ScipyAdapter) and self.getorder() != B.getorder():
      return "sketching scipy matrix to scipy matrix requires same format", None
    if isinstance(B, NumpyAdapter) and self.getorder() != B.getorder():
      return "sketching scipy matrix to numpy matrix requires same format", None
    elif not isinstance(B, NumpyAdapter) and not isinstance(B, ScipyAdapter) and self.getorder() == 'C':
      return "scipy combined with non numpy/scipy must have fortran ordering", None
    else:
      return None, self.getorder() == 'C'

  def getctor(self):
    return ScipyAdapter.ctor

  @staticmethod
  def ctor(m, n, B):
    # Construct scipy matrix that is compatible with B.
    if (isinstance(B, ScipyAdapter) or isinstance(B, NumpyAdapter)) and B.getorder() == 'C':
      return scipy.sparse.csr_matrix((m, n))
    else:
      return scipy.sparse.csc_matrix((m, n))

if ELEM_INSTALLED:
  class DistMatrixAdapter:
    def __init__(self, A):
      self._A = A
      self._dist_data = A.GetDistData()

      if El.TagToType(A.tag) == ctypes.c_double:
        if self._dist_data.colDist == El.MC and self._dist_data.rowDist == El.MR:
          self._ctype = "DistMatrix"
          self._typeid = ""
        else:
          if self._dist_data.colDist == El.CIRC and self._dist_data.rowDist == El.CIRC:
            self._ctype = "SharedMatrix"
          elif self._dist_data.colDist == El.STAR and self._dist_data.rowDist == El.STAR:
            self._ctype = "RootMatrix"
          else:
            tagmap = {El.VC : "VC", El.VR : "VR", El.MC : "MC", El.MR : "MR", El.STAR : "STAR", El.CIRC : "CIRC"}
            self._typeid = tagmap[self._dist_data.colDist] + "_" + tagmap[self._dist_data.rowDist]
            self._ctype = "DistMatrix_" + self._typeid

      elif El.TagToType(A.tag) == ctypes.c_long:
        if self._dist_data.colDist == El.MC and self._dist_data.rowDist == El.MR:
          self._ctype = "DistMatrix_Int"
          self._typeid = ""
        else: 
          raise errors.UnsupportedError("Only MC x MR matrices of Int are supported.")

      else:
        raise errors.UnsupportedError("Only double precision and Int matrices are supported.")

    def ctype(self):
      return self._ctype

    def ptr(self):
      return self._A.obj

    def ptrcleaner(self):
      pass

    def getdim(self, dim):
      if dim == 0:
        return self._A.Height()
      if dim == 1:
        return self._A.Width()

    def getobj(self):
      return self._A

    def iscompatible(self, B):
      if isinstance(B, NumpyAdapter) and B.getorder() != 'F':
        return "numpy combined with other types must have fortran ordering", None
      else:
        return None, False

    def getctor(self):
      return lambda m, n, c : DistMatrixAdapter.ctor(self._dist_data.colDist, self._dist_data.rowDist, m, n, c)

    @staticmethod
    def ctor(coldist, rowdist, m, n, B):
      A = El.DistMatrix(colDist = coldist, rowDist = rowdist)
      A.Resize(m, n)
      return A

  class ElMatrixAdapter:
    def __init__(self, A):
      if El.TagToType(A.tag) != ctypes.c_double:
        raise errors.UnsupportedError("Only double precision matrices are supported.")

      self._A = A

    def ctype(self):
      return "Matrix"

    def ptr(self):
      return self._A.obj

    def ptrcleaner(self):
      pass

    def getdim(self, dim):
      if dim == 0:
        return self._A.Height()
      if dim == 1:
        return self._A.Width()

    def getobj(self):
      return self._A

    def iscompatible(self, B):
      if isinstance(B, NumpyAdapter) and B.getorder() != 'F':
        return "numpy combined with other types must have fortran ordering", None
      else:
        return None, False

    def getctor(self):
      return ElMatrixAdapter.ctor

    @staticmethod
    def ctor(m, n, B):
      A = El.Matrix()
      A.Resize(m, n)
      return A


if KDT_INSTALLED:
  class KDTAdapter:
    def __init__(self, A):
      self._A = A

    def ctype(self):
      return "DistSparseMatrix"

    def ptr(self):
      return ctypes.c_void_p(long(self._A._m_.this))

    def ptrcleaner(self):
      pass

    def getdim(self, dim):
      if dim == 0:
        return self._A.nrow()
      if dim == 1:
        return self._A.ncol()

    def getobj(self):
      return self._A

    def iscompatible(self, B):
      if isinstance(B, NumpyAdapter) and B.getorder() != 'F':
        return "numpy combined with other types must have fortran ordering", None
      else:
        return None, False

    def getctor(self):
      return KDTAdapter.ctor

    @staticmethod
    def ctor(m, n, B):
      import kdt
      nullVec = kdt.Vec(0, sparse=False)
      return kdt.Mat(nullVec, nullVec, nullVec, n, m)

#
# The following functions adapts an object to a uniform interface, so
# that we can have a uniform way of accessing it.
# For convenience it can receive a list and apply the function to each element.
#
def adapt(obj):
    """
    Adapt an object to a uniform interface that can be used to easily pass
    to the C/C++ layers of skylark.
    """
    if ELEM_INSTALLED and sys.modules.has_key('El'):
        global El
        import El
        haselem = True
    else:
        haselem = False

    if KDT_INSTALLED and sys.modules.has_key('kdt'):
        global kdt
        import kdt
        haskdt = True
    else:
        haskdt = False

    if isinstance(obj, list):
      return map(adapt, obj)

    elif isinstance(obj, numpy.ndarray):
        return NumpyAdapter(obj)

    elif isinstance(obj, scipy.sparse.csr_matrix) or isinstance(obj, scipy.sparse.csc_matrix):
        return ScipyAdapter(obj)

    elif haselem and isinstance(obj, El.Matrix):
        return ElMatrixAdapter(obj)

    elif haselem and isinstance(obj, El.DistMatrix):
        return DistMatrixAdapter(obj)

    elif haskdt and isinstance(obj, kdt.Mat):
        return KDTAdapter(obj)

    else:
        raise errors.InvalidObjectError("Invalid/unsupported object passed as parameter")

#
# The following functions call ptrcleaner for each object in the list objects
# Many times after adapt some matrices we have to call the ptrcleaner() function
# This function is used to make our code shorter and more readable
#
def clean_pointer(objects):
    for obj in objects:
        obj.ptrcleaner()

#
# Create mapping between type string and and constructor for that type
#
map_to_ctor = { }
map_to_ctor["LocalMatrix"]   = NumpyAdapter.ctor
map_to_ctor["LocalSpMatrix"] = ScipyAdapter.ctor

if ELEM_INSTALLED:
    map_to_ctor["ElMatrix"] = ElMatrixAdapter.ctor
    map_to_ctor["DistMatrix"] = lambda m, n, c : DistMatrixAdapter.ctor(El.MC, el.MR, m, n, c)
    map_to_ctor["DistMatrix_VR_STAR"] = lambda m, n, c : DistMatrixAdapter.ctor(El.VR, El.STAR, m, n, c)
    map_to_ctor["DistMatrix_VC_STAR"] = lambda m, n, c : DistMatrixAdapter.ctor(El.VC, El.STAR, m, n, c)
    map_to_ctor["DistMatrix_STAR_VR"] = lambda m, n, c : DistMatrixAdapter.ctor(El.STAR, El.VR, m, n, c)
    map_to_ctor["DistMatrix_STAR_VC"] = lambda m, n, c : DistMatrixAdapter.ctor(El.STAR, El.VC, m, n, c)
    map_to_ctor["SharedMatrix"] = lambda m, n, c : DistMatrixAdapter.ctor(El.STAR, El.STAR, m, n, c)
    map_to_ctor["RootMatrix"] = lambda m, n, c : DistMatrixAdapter.ctor(El.CIRC, El.CIRC, m, n, c)

if KDT_INSTALLED:
    map_to_ctor["DistSparseMatrix"] = KDTAdapter.ctor
