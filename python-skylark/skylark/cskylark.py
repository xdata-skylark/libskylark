import errors
import ctypes
from ctypes import byref, cdll, c_double, c_void_p, c_int, c_char_p, pointer, POINTER, c_bool
import math
import numpy
import sys
import os
import time
import atexit

_DEF_OUTTYPE = "LocalMatrix"

# Function for initialization and reinitilialization
def initialize(seed=-1):
  """
  Reinitalize the library with a seed. If seed is -1 then system time is
  used to create the seed.
  """

  global _lib, _ctxt_obj, _ELEM_INSTALLED, _KDT_INSTALLED
  global SUPPORTED_SKETCH_TRANSFORMS
  global _rank, _size

  if '_lib' not in globals():
    try:
      #
      # Load C-API library and set return types
      #
      _lib = cdll.LoadLibrary('libcskylark.so')
      _lib.sl_create_context.restype              = c_int
      _lib.sl_create_default_context.restype      = c_int
      _lib.sl_free_context.restype                = c_int
      _lib.sl_context_rank.restype                = c_int
      _lib.sl_context_size.restype                = c_int
      _lib.sl_create_sketch_transform.restype     = c_int
      _lib.sl_wrap_raw_matrix.restype             = c_int
      _lib.sl_free_raw_matrix_wrap.restype        = c_int
      _lib.sl_strerror.restype                    = c_char_p
      _lib.sl_supported_sketch_transforms.restype = c_char_p
      _lib.sl_has_elemental.restype               = c_bool
      _lib.sl_has_combblas.restype                = c_bool
      
      _ELEM_INSTALLED = _lib.sl_has_elemental()
      _KDT_INSTALLED  = _lib.sl_has_combblas()    
      
      SUPPORTED_SKETCH_TRANSFORMS = map(eval, _lib.sl_supported_sketch_transforms().split())
    except:
      # Did not find library -- must rely on Python code
      _lib = None
      _ELEM_INSTALLED = False
      _KDT_INSTALLED = False
      SUPPORTED_SKETCH_TRANSFORMS = [ ("JLT", "Matrix", "Matrix") ]

    # TODO reload dll ?

  if seed == -1:
    seed = int(time.time())

  if _lib == None:
    # We assume completly local operation when no C++ layer.
    _rank = 1
    _size = 1
    numpy.random.seed(seed)
    return
    
  if '_ctxt_obj' in globals():
    _lib.sl_free_context(_ctxt_obj)

  ctxt_obj = c_void_p()
  _lib.sl_create_default_context(seed, byref(ctxt_obj))
  _ctxt_obj = ctxt_obj.value

  rank = c_int()
  _lib.sl_context_rank(_ctxt_obj, byref(rank))
  _rank = rank.value

  size = c_int()
  _lib.sl_context_size(_ctxt_obj, byref(size))
  _size = size.value

# Allow finalization
def finalize():
  """
  Finalize (de-allocate) the library. However, note that that will not cause
  allocated objects (e.g. sketch transforms) to be freed. They are freed by
  the garbage collector when detected as garbage (no references).
  """
  # TODO free dll (?)
  global _lib, _ctxt_obj, _rank, _size
  if _lib != None:
    if _ctxt_obj != 0:
      _lib.sl_free_context(_ctxt_obj)
    _ctxt_obj = 0

# Make sure finalize is called before exiting (just in case).
atexit.register(finalize)

# Actually initialize the C-API.
initialize(int(time.time()))

# TODO
#def _strerror(errorno):
#  return _lib.sl_strerror(errorno)

#
# Matrix type adapters: specifies how to interact with the underlying (perhaps in C/C++)
# data structure.
#
class _NumpyAdapter:
  def __init__(self, A):
    self._A = A

  def ctype(self):
    return "Matrix"

  def ptr(self):
    if not self._A.flags.f_contiguous:
      raise errors.UnsupportedError("Only FORTRAN style (column-major) NumPy arrays are supported")
    else:
      data = c_void_p()
      _lib.sl_wrap_raw_matrix( \
        self._A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
        self._A.shape[0], self._A.shape[1] if self._A.ndim > 1 else 1 , byref(data))
      self._ptr = data.value
      return data.value

  def ptrcleaner(self):
    _lib.sl_free_raw_matrix_wrap(self._ptr);

  def getdim(self, dim):
    return self._A.shape[dim]

  def getobj(self):
    return self._A

  @staticmethod
  def ctor(m, n):
    # C++ layer works only with Fortran ordering, but in a pure Python
    # implementation we default to C ordering
    if _lib != None:
      return numpy.empty((m,n), order='F')
    else:
      return numpy.empty((m,n))

if _ELEM_INSTALLED:
  class _ElemAdapter:
    def __init__(self, A):
      self._A = A
      if isinstance(A, elem.DistMatrix_d):
        self._typestr = "DistMatrix"
      elif isinstance(A, elem.DistMatrix_d_VC_STAR):
        self._typestr = "DistMatrix_VC_STAR"
      elif isinstance(A, elem.DistMatrix_d_VR_STAR):
        self._typestr = "DistMatrix_VR_STAR"
      elif isinstance(A, elem.DistMatrix_d_STAR_VC):
        self._typestr = "DistMatrix_STAR_VC"
      elif isinstance(A, elem.DistMatrix_d_STAR_VR):
        self._typestr = "DistMatrix_STAR_VR"
      else:
        raise errors.UnsupportedError("Unsupported Elemental type")

    def ctype(self):
      return self._typestr

    def ptr(self):
      return ctypes.c_void_p(long(self._A.this))

    def ptrcleaner(self):
      None

    def getdim(self, dim):
      if dim == 0:
        return self._A.Height
      if dim == 1:
        return self._A.Width

    def getobj(self):
      return self._A

    @staticmethod
    def ctor(typestr, m, n):
      if typestr == "":
        cls = elem.DistMatrix_d
      else:
        cls = eval("elem.DistMatrix_d_" + typestr)
      cls(m, n)


if _KDT_INSTALLED:
  class _KDTAdapter:
    def __init__(self, A):
      self._A = A

    def ctype(self):
      return "DistSparseMatrix"

    def ptr(self):
      return ctypes.c_void_p(long(self._A._m_.this))

    def ptrcleaner(self):
      None

    def getdim(self, A, dim):
      if dim == 0:
        return self._A.nrow()
      if dim == 1:
        return self._A.ncol()

    def getobj(self):
      return self._A

    @staticmethod
    def ctor(m, n):
      import kdt
      nullVec = kdt.Vec(0, sparse=False)
      return kdt.Mat(nullVec, nullVec, nullVec, n, m)

#
# The following functions adapts an object to a uniform interface, so 
# that we can have a uniform way of accessing it. 
#
def _adapt(obj):
  if _ELEM_INSTALLED and sys.modules.has_key('elem'):
    global elem
    import elem
    elemcls = [elem.DistMatrix_d,
               elem.DistMatrix_d_VR_STAR, elem.DistMatrix_d_VC_STAR, 
               elem.DistMatrix_d_STAR_VC, elem.DistMatrix_d_STAR_VR] 
  else:
    elemcls = []

  if _KDT_INSTALLED and sys.modules.has_key('kdt'):
    global kdt
    import kdt
    kdtcls = [kdt.Mat]
  else:
    kdtcls = [];

  if isinstance(obj, numpy.ndarray):
    return _NumpyAdapter(obj)

  elif any(isinstance(obj, c) for c in elemcls):
    return _ElemAdapter(obj)

  elif any(isinstance(obj, c) for c in kdtcls):
      return _KDTAdapter(obj)
  
  else:
    raise errors.InvalidObjectError("Invalid/unsupported object passed as A or SA")

#
# Create mapping between type string and and constructor for that type
#
_map_to_ctor = { }
_map_to_ctor["LocalMatrix"] = _NumpyAdapter.ctor

if _ELEM_INSTALLED:
  _map_to_ctor["DistMatrix"] = lambda m, n : _ElemAdapter.ctor("", m, n)
  _map_to_ctor["DistMatrix_VR_STAR"] = lambda m, n : _ElemAdapter.ctor("VR_STAR", m, n)
  _map_to_ctor["DistMatrix_VC_STAR"] = lambda m, n : _ElemAdapter.ctor("VC_STAR", m, n)
  _map_to_ctor["DistMatrix_STAR_VR"] = lambda m, n : _ElemAdapter.ctor("STAR_VC", m, n)
  _map_to_ctor["DistMatrix_STAR_VC"] = lambda m, n : _ElemAdapter.ctor("STAR_VR", m, n)

if _KDT_INSTALLED:
  _map_to_ctor["DistSparseMatrix"] = _KDTAdapter.ctor

#
# Generic Sketch Transform
#
class _SketchTransform(object):
  """
  Base class sketch transforms.
  The various sketch transforms derive from this class and
  which holds the common interface. Derived classes can have different constructors.
  """
  def __init__(self, ttype, n, s, defouttype):
    self._baseinit(ttype, n, s, defouttype)
    if _lib != None:
      sketch_transform = c_void_p()
      _lib.sl_create_sketch_transform(_ctxt_obj, ttype, n, s, byref(sketch_transform))
      self._obj = sketch_transform.value

  def _baseinit(self, ttype, n, s, defouttype):
    if not _map_to_ctor.has_key(defouttype):
      raise errors.UnsupportedError("Unsupported default output type (%s)" % intype)
    self._ttype = ttype
    self._n = n
    self._s = s
    self._defouttype = defouttype

  def __del__(self):
    if _lib != None:
      _lib.sl_free_sketch_transform(self._obj)

  def apply(self, A, SA, dim=0):
    """
    Apply the transform on **A** along dimension **dim** and write
    result in **SA**. Note: for rowwise (aka right) sketching A
    is mapped to A * S^T.

    :param A: Input matrix.
    :param SA: Ouptut matrix. If "None" then the output will be allocated.
    :param dim: Dimension to apply along. 0 - columnwise, 1 - rowwise.
                or can use "columnwise"/"rowwise", "left"/"right"
                default is columnwise

    :return SA
    """
    if dim == 0 or dim == "columnwise" or dim == "left":
      dim = 0
    if dim == "rowwise" or dim == "right":
      dim = 1
    if dim != 0 and dim != 1:
      raise ValueError("Dimension must be either columnwise/rowwise or left/right or 0/1")

    A = _adapt(A)

    # Allocate in case SA is not given, and then adapt it.
    if SA is None:
      ctor = _map_to_ctor[self._defouttype]
      if dim == 0:
        SA = ctor(self._s, A.getdim(1))
      if dim == 1:
        SA = ctor(A.getdim(0), self._s)
    SA = _adapt(SA)

    reqcomb = (self._ttype, A.ctype(), SA.ctype())
    if reqcomb not in SUPPORTED_SKETCH_TRANSFORMS:
      raise errors.UnsupportedError("Unsupported transform-input-output combination: " + str(reqcomb))  

    if A.getdim(dim) != self._n:
      raise errors.DimensionMistmatchError("Sketched dimension is incorrect (input)")
    if SA.getdim(dim) != self._s:
      raise errors.DimensionMistmatchError("Sketched dimension is incorrect (output)")
    if A.getdim(1 - dim) != SA.getdim(1 - dim):
      raise errors.DimensionMistmatchError("Sketched dimension is incorrect (input != output)")


    if _lib != None:
      Aobj = A.ptr()
      SAobj = SA.ptr()
      if (Aobj == -1 or SAobj == -1):
        raise errors.InvalidObjectError("Invalid/unsupported object passed as A or SA")

      _lib.sl_apply_sketch_transform(self._obj, \
                                       A.ctype(), Aobj, SA.ctype(), SAobj, dim+1)

      A.ptrcleaner()
      SA.ptrcleaner()

    else:
      self.ppyapply(A.getobj(), SA.getobj(), dim)

    return SA.getobj()

  def __mul__(self, A):
    return self.apply(A, None, dim=0)

  def __div__(self, A):
    return self.apply(A, None, dim=1)


#
# Various sketch transforms
#

class JLT(_SketchTransform):
  """
  Johnson-Lindenstrauss Transform
  """
  def __init__(self, n, s, outtype=_DEF_OUTTYPE):
    super(JLT, self).__init__("JLT", n, s, outtype);
    if _lib == None:
      # The following is not memory efficient, but for a pure Python impl it will do
      self.S = numpy.matrix(numpy.random.randn(s, n) / math.sqrt(s))

  def ppyapply(self, A, SA, dim):
    if dim == 0:
      SA1 = numpy.dot(self.S, A)
    if dim == 1:
      SA1 = numpy.dot(A, self.S.T)

    # We really want to use the out parameter of numpy.dot, but it does not seem 
    # to work (raises a ValueError)
    numpy.copyto(SA, SA1)

class CT(_SketchTransform):
  """
  Cauchy Transform
  """
  def __init__(self, n, s, C, outtype=_DEF_OUTTYPE):
    super(CT, self)._baseinit("CT", n, s, outtype)

    if _lib != None:
      sketch_transform = c_void_p()
      _lib.sl_create_sketch_transform(_ctxt_obj, "CT", n, s, \
                                        byref(sketch_transform), ctypes.c_double(C))
      self._obj = sketch_transform.value

class FJLT(_SketchTransform):
  """
  Fast Johnson-Lindenstrauss Transform
  """
  def __init__(self, n, s, outtype=_DEF_OUTTYPE):
    super(FJLT, self).__init__("FJLT", n, s, outtype);

class CWT(_SketchTransform):
  """
  Clarkson-Woodruff Transform (also known as CountSketch)

  *K. Clarkson* and *D. Woodruff*, **Low Rank Approximation and Regression
  in Input Sparsity Time**, STOC 2013
  """
  def __init__(self, n, s, outtype=_DEF_OUTTYPE):
    super(CWT, self).__init__("CWT", n, s, outtype);

class MMT(_SketchTransform):
  """
  Meng-Mahoney Transform

  *X. Meng* and *M. W. Mahoney*, **Low-distortion Subspace Embeddings in
  Input-sparsity Time and Applications to Robust Linear Regression**, STOC 2013
  """
  def __init__(self, n, s, outtype=_DEF_OUTTYPE):
    super(MMT, self).__init__("MMT", n, s, intype, outtype);

class WZT(_SketchTransform):
  """
  Woodruff-Zhang Transform

  *D. Woodruff* and *Q. Zhang*, **Subspace Embeddings and L_p Regression
  Using Exponential Random**, COLT 2013
  """
  def __init__(self, n, s, p, outtype=_DEF_OUTTYPE):
    super(WZT, self)._baseinit("WZT", n, s, outtype)

    if _lib != None:
      sketch_transform = c_void_p()
      _lib.sl_create_sketch_transform(_ctxt_obj, "WZT", n, s, \
                                        byref(sketch_transform), ctypes.c_double(p))
      self._obj = sketch_transform.value

class GaussianRFT(_SketchTransform):
  """
  Random Features Transform for the RBF Kernel
  """
  def __init__(self, n, s, sigma, outtype=_DEF_OUTTYPE):
    super(GaussianRFT, self)._baseinit("GaussianRFT", n, s, outtype)

    if _lib != None:
      sketch_transform = c_void_p()
      _lib.sl_create_sketch_transform(_ctxt_obj, "GaussianRFT", n, s, \
                                        byref(sketch_transform), ctypes.c_double(sigma))
      self._obj = sketch_transform.value

class LaplacianRFT(_SketchTransform):
  """
  Random Features Transform for the Laplacian Kernel

  *A. Rahimi* and *B. Recht*, **Random Features for Large-scale
  Kernel Machines*, NIPS 2009
  """
  def __init__(self, n, s, sigma, outtype=_DEF_OUTTYPE):
    super(LaplacianRFT, self)._baseinit("LaplacianRFT", n, s, outtype)

    if _lib != None:
      sketch_transform = c_void_p()
      _lib.sl_create_sketch_transform(_ctxt_obj, "LaplacianRFT", n, s, \
                                        byref(sketch_transform), ctypes.c_double(sigma))
      self._obj = sketch_transform.value
