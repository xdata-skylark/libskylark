import ctypes
from ctypes import byref, cdll, c_double, c_void_p, c_int, pointer, POINTER
import numpy
import sys
import os
import time
import atexit

# TODO: Get these from outside
_ELEM_INSTALLED = True
_KDT_INSTALLED = True

if _ELEM_INSTALLED:
  import elem

if _KDT_INSTALLED:
  import kdt

_DEF_INTYPE = "LocalMatrix"
_DEF_OUTTYPE = "LocalMatrix"

#
# Load C-API library and set return types
#
_lib = cdll.LoadLibrary('libcskylark.so')
_lib.sl_create_context.restype = c_void_p
_lib.sl_create_default_context.restype = c_void_p
_lib.sl_context_rank.restype = c_int
_lib.sl_context_size.restype = c_int
_lib.sl_create_sketch_transform.restype = c_void_p
_lib.sl_wrap_raw_matrix.restype = c_void_p

#
# Create mapping between type string that can be supplied by user
# to one the C-API recognaizes
#
_map_to_ctype = { }
if _ELEM_INSTALLED:
  _map_to_ctype["DistMatrix_VR_STAR"] = "DistMatrix_VR_STAR"
  _map_to_ctype["DistMatrix_VC_STAR"] = "DistMatrix_VC_STAR"
  _map_to_ctype["DistMatrix_STAR_VR"] = "DistMatrix_STAR_VR"
  _map_to_ctype["DistMatrix_STAR_VC"] = "DistMatrix_STAR_VC"

if _KDT_INSTALLED:
  _map_to_ctype["DistSparseMatrix"] = "DistSparseMatrix"

_map_to_ctype["LocalMatrix"] = "Matrix"

# 
# Create mapping between object type to function converting it 
# to pointers to passed to the C-API
#
def _elem_to_ptr(A):
  return ctypes.c_void_p(long(A.this))

def _kdt_to_ptr(A):
  return ctypes.c_void_p(long(A._m_.this))

def _np_to_ptr(A):
  if not A.flags.f_contiguous:
    if _rank == 0:
      print "ERROR: only FORTRAN style (column-major) NumPy arrays are supported" # TODO
    return -1
  else:
    return _lib.sl_wrap_raw_matrix( \
      A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
        A.shape[0], A.shape[1] if A.ndim > 1 else 1)

def _np_ptr_cleaner(p):
  _lib.sl_free_raw_matrix_wrap(p);

# TODO: classify the following

_map_to_ptr = { }
if _ELEM_INSTALLED:
  _map_to_ptr["DistMatrix_VR_STAR"] = _elem_to_ptr
  _map_to_ptr["DistMatrix_VC_STAR"] = _elem_to_ptr
  _map_to_ptr["DistMatrix_STAR_VR"] = _elem_to_ptr
  _map_to_ptr["DistMatrix_STAR_VC"] = _elem_to_ptr

if _KDT_INSTALLED:
  _map_to_ptr["DistSparseMatrix"] = _kdt_to_ptr

_map_to_ptr["LocalMatrix"] = _np_to_ptr

_map_to_ptr_cleaner = { }
_map_to_ptr_cleaner["LocalMatrix"]  = _np_ptr_cleaner

def _elem_ctor(etype):
  return lambda m,n : etype(m, n)

def _kdt_ctor(m, n) :
  nullVec = kdt.Vec(0, sparse=False)
  return kdt.Math(nullVec, nullVec, nullVec, n, m)

_map_to_ctor = { }
if _ELEM_INSTALLED:
  _map_to_ctor["DistMatrix_VR_STAR"] = _elem_ctor(elem.DistMatrix_d_VR_STAR)
  _map_to_ctor["DistMatrix_VC_STAR"] = _elem_ctor(elem.DistMatrix_d_VC_STAR)
  _map_to_ctor["DistMatrix_STAR_VR"] = _elem_ctor(elem.DistMatrix_d_STAR_VR)
  _map_to_ctor["DistMatrix_STAR_VC"] = _elem_ctor(elem.DistMatrix_d_STAR_VC)
if _KDT_INSTALLED:
  _map_to_ctor["DistSparseMatrix"] = _kdt_ctor
_map_to_ctor["LocalMatrix"] = lambda m, n: numpy.empty((m,n), order='F')

def _elem_getdim(A, dim):
  if dim == 0:
    return A.Height
  if dim == 1:
    return A.Width

def _kdt_getdim(A, dim):
  if dim == 0:
    return A.nrow()
  if dim == 1:
    return A.ncol()

def _np_getdim(A, dim):
  return A.shape[dim]

_map_to_getdim = { }
if _ELEM_INSTALLED:
  _map_to_getdim["DistMatrix_VR_STAR"] = _elem_getdim
  _map_to_getdim["DistMatrix_VC_STAR"] = _elem_getdim
  _map_to_getdim["DistMatrix_STAR_VR"] = _elem_getdim
  _map_to_getdim["DistMatrix_STAR_VC"] = _elem_getdim
if _KDT_INSTALLED:
  _map_to_getdim["DistSparseMatrix"] = _kdt_getdim
_map_to_getdim["LocalMatrix"] = lambda A,dim : A.shape[dim]

# Function for initialization and reinitilialization
def initialize(seed=-1):
  """
  Reinitalize the library with a seed. If seed is -1 then system time is
  used to create the seed.
  """
  # TODO reload dll (?)
  global _ctxt_obj, _rank, _size
  if seed == -1:
    seed = int(time.time())
  if _ctxt_obj != 0:
    _lib.sl_free_context(_ctxt_obj)
  _ctxt_obj = _lib.sl_create_default_context(seed)
  _rank = _lib.sl_context_rank(_ctxt_obj)
  _size = _lib.sl_context_size(_ctxt_obj)

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
# Generic Sketch Transform
#
class SketchTransform(object):
  """
  Base class sketch transforms.
  The various sketch transforms derive from this class and 
  which holds the common interface. Derived classes can have different constructors.
  """
  def __init__(self, ttype, n, s, intype, outtype):
    self._baseinit(n, s, intype, outtype)
    self._obj = _lib.sl_create_sketch_transform(_ctxt_obj, ttype, \
                                                _map_to_ctype[intype], \
                                                _map_to_ctype[outtype], n, s)

  def _baseinit(self, n, s, intype, outtype):
    if not _map_to_ctype.has_key(intype):
      if _rank == 0:
        print "ERROR: unknown input type (%s)" % intype      # TODO
      return -1

    if not _map_to_ctype.has_key(outtype):
      if _rank == 0:
        print "ERROR: unknown output type (%s)" % outtype    # TODO
      return -1

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
    if dim == "columnwise" or dim == "left":
      dim = 0
    if dim == "rowwise" or dim == "right":
      dim = 1

    if SA == None:
      ctor = _map_to_ctor[self._outtype];
      getdim = _map_to_getdim[self._intype];
      if dim == 0:
        SA = ctor(self._s, getdim(A, 1))
      if dim == 1:
        SA = ctor(getdim(A, 0), self._s)

    Aobj = _map_to_ptr[self._intype](A)
    SAobj = _map_to_ptr[self._outtype](SA)
    if (Aobj == -1 or SAobj == -1):
      return -1

    _lib.sl_apply_sketch_transform(self._obj, Aobj, SAobj, dim+1)

    if _map_to_ptr_cleaner.has_key(self._intype):
      _map_to_ptr_cleaner[self._intype](Aobj)

    if _map_to_ptr_cleaner.has_key(self._outtype):
      _map_to_ptr_cleaner[self._outtype](SAobj)

    return SA

  def __mul__(self, A):
    return self.apply(A, None, dim=0)

  def __div__(self, A):
    return self.apply(A, None, dim=1)

#
# Various sketch transforms
#

class JLT(SketchTransform):
  """
  Johnson-Lindenstrauss Transform
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(JLT, self).__init__("JLT", n, s, intype, outtype);

class CT(SketchTransform):
  """
  Cauchy Transform
  """
  def __init__(self, n, s, C, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(CT, self)._baseinit(n, s, intype, outtype)
    self._obj = _lib.sl_create_sketch_transform(_ctxt_obj, "CT", \
                                                _map_to_ctype[intype], \
                                                _map_to_ctype[outtype], n, s, ctypes.c_double(C))

class FJLT(SketchTransform):
  """
  Fast Johnson-Lindenstrauss Transform
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(FJLT, self).__init__("FJLT", n, s, intype, outtype);

class CWT(SketchTransform):
  """
  Clarkson-Woodruff Transform (also known as CountSketch)

  *K. Clarkson* and *D. Woodruff*, **Low Rank Approximation and Regression
  in Input Sparsity Time**, STOC 2013 
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(CWT, self).__init__("CWT", n, s, intype, outtype);

class MMT(SketchTransform):
  """
  Meng-Mahoney Transform

  *X. Meng* and *M. W. Mahoney*, **Low-distortion Subspace Embeddings in
  Input-sparsity Time and Applications to Robust Linear Regression**, STOC 2013
  """
  def __init__(self, n, s, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(MMT, self).__init__("MMT", n, s, intype, outtype);

class WZT(SketchTransform):
  """
  Woodruff-Zhang Transform

  *D. Woodruff* and *Q. Zhang*, **Subspace Embeddings and L_p Regression
  Using Exponential Random**, COLT 2013
  """
  def __init__(self, n, s, p, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(WZT, self)._baseinit(n, s, intype, outtype)
    self._obj = _lib.sl_create_sketch_transform(_ctxt_obj, "WZT", \
                                                _map_to_ctype[intype], \
                                                _map_to_ctype[outtype], n, s, ctypes.c_double(p))

class GaussianRFT(SketchTransform):
  """
  Random Features Transform for the RBF Kernel
  """
  def __init__(self, n, s, sigma, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(GaussianRFT, self)._baseinit(n, s, intype, outtype)
    self._obj = _lib.sl_create_sketch_transform(_ctxt_obj, "GaussianRFT", \
                                                _map_to_ctype[intype], \
                                                _map_to_ctype[outtype], n, s, ctypes.c_double(sigma))

class LaplacianRFT(SketchTransform):
  """
  Random Features Transform for the Laplacian Kernel

  *A. Rahimi* and *B. Recht*, **Random Features for Large-scale 
  Kernel Machines*, NIPS 2009
  """
  def __init__(self, n, s, sigma, intype=_DEF_INTYPE, outtype=_DEF_OUTTYPE):
    super(LaplacianRFT, self)._baseinit(n, s, intype, outtype)
    self._obj = _lib.sl_create_sketch_transform(_ctxt_obj, "LaplacianRFT", \
                                                _map_to_ctype[intype], \
                                                _map_to_ctype[outtype], n, s, ctypes.c_double(sigma))

