import errors
import ctypes
from ctypes import byref, cdll, c_double, c_void_p, c_int, c_char_p, pointer, POINTER, c_bool
import numpy
import sys
import os
import time
import atexit

_DEF_INTYPE  = "LocalMatrix"
_DEF_OUTTYPE = "LocalMatrix"

#
# Load C-API library and set return types
#
_lib = cdll.LoadLibrary('libcskylark.so')
_lib.sl_create_context.restype          = c_int
_lib.sl_create_default_context.restype  = c_int
_lib.sl_free_context.restype            = c_int
_lib.sl_context_rank.restype            = c_int
_lib.sl_context_size.restype            = c_int
_lib.sl_create_sketch_transform.restype = c_int
_lib.sl_wrap_raw_matrix.restype         = c_int
_lib.sl_free_raw_matrix_wrap.restype    = c_int

_lib.sl_strerror.restype                    = c_char_p
_lib.sl_supported_sketch_transforms.restype = c_char_p

_lib.sl_has_elemental.restype = c_bool
_lib.sl_has_combblas.restype  = c_bool

SUPPORTED_SKETCH_TRANSFORMS = map(eval, _lib.sl_supported_sketch_transforms().split())

_ELEM_INSTALLED = _lib.sl_has_elemental()
_KDT_INSTALLED  = _lib.sl_has_combblas()

#
# Simple helper to convert error codes in human readbale strings
#
def _strerror(errorno):
  return _lib.sl_strerror(errorno)

#
# Matrix type adapters: specifies how to interact with the underlying (in C/C++)
# data structure.
#
class _NumpyAdapter:
  def ctype(self):
    return "Matrix"

  def ctor(self, m, n):
    return numpy.empty((m,n), order='F')

  def ptr(self, A):
    if not A.flags.f_contiguous:
      raise errors.UnsupportedError("Only FORTRAN style (column-major) NumPy arrays are supported")
    else:
      data = c_void_p()
      _lib.sl_wrap_raw_matrix( \
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
        A.shape[0], A.shape[1] if A.ndim > 1 else 1 , byref(data))
      return data.value

  def ptr_cleaner(self, p):
    _lib.sl_free_raw_matrix_wrap(p);

  def getdim(self, A, dim):
    return A.shape[dim]

if _ELEM_INSTALLED:
  class _ElemAdapter:
    def __init__(self, typestr):
      import elem
      self._typestr = "DistMatrix_" + typestr
      self._class = eval("elem.DistMatrix_d_" + typestr)

    def ctype(self):
      return self._typestr

    def ctor(self, m, n):
      return self._class(m, n)

    def ptr(self, A):
      return ctypes.c_void_p(long(A.this))

    def ptr_cleaner(self, p):
      None

    def getdim(self, A, dim):
      if dim == 0:
        return A.Height
      if dim == 1:
        return A.Width

if _KDT_INSTALLED:
  class _KDTAdapter:
    def ctype(self):
      return "DistSparseMatrix"

    def ctor(self, m, n):
      import kdt
      nullVec = kdt.Vec(0, sparse=False)
      return kdt.Mat(nullVec, nullVec, nullVec, n, m)

    def ptr(self, A):
      return ctypes.c_void_p(long(A._m_.this))

    def ptr_cleaner(self, p):
      None

    def getdim(self, A, dim):
      if dim == 0:
        return A.nrow()
      if dim == 1:
        return A.ncol()

#
# Create mapping between type string and matrix type adapter
_map_to_adapter = { }
_map_to_adapter["LocalMatrix"] = _NumpyAdapter()

if _ELEM_INSTALLED:
  _map_to_adapter["DistMatrix_VR_STAR"] = _ElemAdapter("VR_STAR")
  _map_to_adapter["DistMatrix_VC_STAR"] = _ElemAdapter("VC_STAR")
  _map_to_adapter["DistMatrix_STAR_VR"] = _ElemAdapter("STAR_VR")
  _map_to_adapter["DistMatrix_STAR_VC"] = _ElemAdapter("STAR_VC")

if _KDT_INSTALLED:
  _map_to_adapter["DistSparseMatrix"] = _KDTAdapter()

# Function for initialization and reinitilialization
def initialize(seed=-1):
  """
  Reinitalize the library with a seed. If seed is -1 then system time is
  used to create the seed.
  """
  # TODO reload dll (?)
  global _ctxt_obj
  if seed == -1:
    seed = int(time.time())
  if _ctxt_obj != 0:
    _lib.sl_free_context(_ctxt_obj)

  ctxt_obj = c_void_p()
  _lib.sl_create_default_context(seed, byref(ctxt_obj))
  _ctxt_obj = ctxt_obj.value

  global _rank
  rank = c_int()
  _lib.sl_context_rank(_ctxt_obj, byref(rank))
  _rank = rank.value

  global _size
  size = c_int()
  _lib.sl_context_size(_ctxt_obj, byref(size))
  _size = size.value


# Actually initialize the C-API.
_ctxt_obj = 0
initialize(int(time.time()))

# Allow finalization
def finalize():
  """
  Finalize (de-allocate) the library. However, note that that will not cause
  allocated objects (e.g. sketch transforms) to be freed. They are freed by
  the garbage collector when detected as garbage (no references).
  """
  # TODO free dll (?)
  global _ctxt_obj, _rank, _size
  if _ctxt_obj != 0:
    _lib.sl_free_context(_ctxt_obj)
  _ctxt_obj = 0

# Make sure finalize is called before exiting (just in case).
atexit.register(finalize)

#
#
# Generic Sketch Transform
#
class _SketchTransform(object):
  """
  Base class sketch transforms.
  The various sketch transforms derive from this class and
  which holds the common interface. Derived classes can have different constructors.
  """
  def __init__(self, ttype, n, s, intype, outtype):
    reqcomb = (ttype, \
               _map_to_adapter[intype].ctype(), \
               _map_to_adapter[outtype].ctype())

    if reqcomb not in SUPPORTED_SKETCH_TRANSFORMS:
      raise errors.UnsupportedError("Unsupported sketch transofrm " + str(reqcomb))

    sketch_transform = c_void_p()
    self._baseinit(n, s, intype, outtype)

    _lib.sl_create_sketch_transform(_ctxt_obj, ttype, \
                                    _map_to_adapter[intype].ctype(), \
                                    _map_to_adapter[outtype].ctype(), n, s, \
                                    byref(sketch_transform))

    self._obj = sketch_transform.value


  def _baseinit(self, n, s, intype, outtype):
    if not _map_to_adapter.has_key(intype):
      raise errors.UnsupportedError("Unsupported input type (%s)" % intype)

    if not _map_to_adapter.has_key(outtype):
      raise errors.UnsupportedError("Unsupported input type (%s)" % intype)

    self._intype = intype
    self._outtype = outtype
    self._n = n
    self._s = s

  def __del__(self):
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

    # Allocate in case SA is not given
    if SA is None:
      ctor = _map_to_adapter[self._outtype].ctor
      getdim = _map_to_adapter[self._intype].getdim
      if dim == 0:
        SA = ctor(self._s, getdim(A, 1))
      if dim == 1:
        SA = ctor(getdim(A, 0), self._s)

    # Verify dimensions
    ingetdim = _map_to_adapter[self._intype].getdim
    outgetdim = _map_to_adapter[self._outtype].getdim
    if ingetdim(A, dim) != self._n:
      raise errors.DimensionMistmatchError("Sketched dimension is incorrect (input)")
    if outgetdim(SA, dim) != self._s:
      raise errors.DimensionMistmatchError("Sketched dimension is incorrect (output)")
    if ingetdim(A, 1 - dim) != outgetdim(SA, 1 - dim):
      raise errors.DimensionMistmatchError("Sketched dimension is incorrect (output)")

    Aobj = _map_to_adapter[self._intype].ptr(A)
    SAobj = _map_to_adapter[self._outtype].ptr(SA)
    if (Aobj == -1 or SAobj == -1):
      return -1

    _lib.sl_apply_sketch_transform(self._obj, Aobj, SAobj, dim+1)

    _map_to_adapter[self._intype].ptr_cleaner(Aobj)
    _map_to_adapter[self._outtype].ptr_cleaner(SAobj)

    return SA

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
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(JLT, self).__init__("JLT", n, s, intype, outtype);

class CT(_SketchTransform):
  """
  Cauchy Transform
  """
  def __init__(self, n, s, C, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(CT, self)._baseinit(n, s, intype, outtype)

    sketch_transform = c_void_p()
    _lib.sl_create_sketch_transform(_ctxt_obj, "CT", \
                                    _map_to_adapter[intype].ctype(), \
                                    _map_to_adapter[outtype].ctype(), n, s, \
                                    byref(sketch_transform), ctypes.c_double(C))
    self._obj = sketch_transform.value

class FJLT(_SketchTransform):
  """
  Fast Johnson-Lindenstrauss Transform
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(FJLT, self).__init__("FJLT", n, s, intype, outtype);

class CWT(_SketchTransform):
  """
  Clarkson-Woodruff Transform (also known as CountSketch)

  *K. Clarkson* and *D. Woodruff*, **Low Rank Approximation and Regression
  in Input Sparsity Time**, STOC 2013
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(CWT, self).__init__("CWT", n, s, intype, outtype);

class MMT(_SketchTransform):
  """
  Meng-Mahoney Transform

  *X. Meng* and *M. W. Mahoney*, **Low-distortion Subspace Embeddings in
  Input-sparsity Time and Applications to Robust Linear Regression**, STOC 2013
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(MMT, self).__init__("MMT", n, s, intype, outtype);

class WZT(_SketchTransform):
  """
  Woodruff-Zhang Transform

  *D. Woodruff* and *Q. Zhang*, **Subspace Embeddings and L_p Regression
  Using Exponential Random**, COLT 2013
  """
  def __init__(self, n, s, p, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(WZT, self)._baseinit(n, s, intype, outtype)

    sketch_transform = c_void_p()
    _lib.sl_create_sketch_transform(_ctxt_obj, "WZT", \
                                    _map_to_adapter[intype].ctype(), \
                                    _map_to_adapter[outtype].ctype(), n, s, \
                                    byref(sketch_transform), ctypes.c_double(p))
    self._obj = sketch_transform.value

class GaussianRFT(_SketchTransform):
  """
  Random Features Transform for the RBF Kernel
  """
  def __init__(self, n, s, sigma, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(GaussianRFT, self)._baseinit(n, s, intype, outtype)

    sketch_transform = c_void_p()
    _lib.sl_create_sketch_transform(_ctxt_obj, "GaussianRFT", \
                                    _map_to_adapter[intype].ctype(), \
                                    _map_to_adapter[outtype].ctype(), n, s, \
                                    byref(sketch_transform), ctypes.c_double(sigma))
    self._obj = sketch_transform.value

class LaplacianRFT(_SketchTransform):
  """
  Random Features Transform for the Laplacian Kernel

  *A. Rahimi* and *B. Recht*, **Random Features for Large-scale
  Kernel Machines*, NIPS 2009
  """
  def __init__(self, n, s, sigma, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(LaplacianRFT, self)._baseinit(n, s, intype, outtype)

    sketch_transform = c_void_p()
    _lib.sl_create_sketch_transform(_ctxt_obj, "LaplacianRFT", \
                                    _map_to_adapter[intype].ctype(), \
                                    _map_to_adapter[outtype].ctype(), n, s, \
                                    byref(sketch_transform), ctypes.c_double(sigma))
    self._obj = sketch_transform.value

