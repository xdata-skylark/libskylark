import ctypes
from ctypes import byref, cdll, c_double, c_void_p, c_int, pointer, POINTER
import numpy
import elem 
import sys
import os

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
_map_to_ctype["DistMatrix_VR_STAR"] = "DistMatrix_VR_STAR"
_map_to_ctype["DistMatrix_VC_STAR"] = "DistMatrix_VC_STAR"
_map_to_ctype["DistMatrix_STAR_VR"] = "DistMatrix_STAR_VR"
_map_to_ctype["DistMatrix_STAR_VC"] = "DistMatrix_STAR_VC"
_map_to_ctype["DistSparseMatrix"] = "DistSparseMatrix"
_map_to_ctype["LocalMatrix"] = "Matrix"


# 
# Create mapping between object type to function converting it 
# to pointers to passed to the C-API
#
def _elem_to_ptr(ctxt, A):
  return ctypes.c_void_p(long(A.this))

def _kdt_to_ptr(ctxt, A):
  return ctypes.c_void_p(long(A._m_.this))

def _np_to_ptr(ctxt, A):
  if not A.flags.f_contiguous:
    if ctxt.rank() == 0:
      print "ERROR: only FORTRAN style (column-major) NumPy arrays are supported" # TODO
    return -1
  else:
    return _lib.sl_wrap_raw_matrix( \
      A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), \
        A.shape[0], A.shape[1])

def _np_ptr_cleaner(p):
  _lib.sl_free_raw_matrix_wrap(p);

_map_to_ptr = { }
_map_to_ptr["DistMatrix_VR_STAR"] = _elem_to_ptr
_map_to_ptr["DistMatrix_VC_STAR"] = _elem_to_ptr
_map_to_ptr["DistMatrix_STAR_VR"] = _elem_to_ptr
_map_to_ptr["DistMatrix_STAR_VC"] = _elem_to_ptr
_map_to_ptr["DistSparseMatrix"] = _kdt_to_ptr
_map_to_ptr["LocalMatrix"] = _np_to_ptr

_map_to_ptr_cleaner = { }
_map_to_ptr_cleaner["LocalMatrix"]  = _np_ptr_cleaner

#
# Context
#
class Context(object):
  """
  Create a Skylark Context Object
  """
  def __init__(self, seed):
    self._obj = _lib.sl_create_default_context(seed)
  # TODO: Figure out how to wrap MPI_Comm

  def free(self):
    _lib.sl_free_context(self._obj)
    del self._obj

  def size(self):
    return _lib.sl_context_size(self._obj)

  def rank(self):
    return _lib.sl_context_rank(self._obj)

#
# Generic Sketch Transform
#
class SketchTransform(object):
  """
  Base class sketch transforms.
  The various sketch transforms derive from this class and 
  which holds the common interface. Derived classes can have different constructors.
  """
  def __init__(self, ctxt, ttype, intype, outtype, n, s):
    if not _map_to_ctype.has_key(intype):
      if ctxt.rank() == 0:
        print "ERROR: unknown input type (%s)" % intype      # TODO
      return -1

    if not _map_to_ctype.has_key(outtype):
      if ctxt.rank() == 0:
        print "ERROR: unknown output type (%s)" % outtype    # TODO
      return -1

    self._ctxt = ctxt;
    self._intype = intype
    self._outtype = outtype
    self._obj = _lib.sl_create_sketch_transform(ctxt._obj, ttype, \
                                                _map_to_ctype[intype], \
                                                _map_to_ctype[outtype], n, s)

  def free(self):
    """ Discard the transform """
    _lib.sl_free_sketch_transform(self._obj)
    del self._obj

  def apply(self, A, SA, dim):
    """
    Apply the transform on **A** along dimension **dim** and write
    result in **SA**.

    :param A: Input matrix.
    :param SA: Ouptut matrix.
    :param dim: Dimension to apply along. 1 - columnwise, 2 - rowwise.

    """
    Aobj = _map_to_ptr[self._intype](self._ctxt, A)
    SAobj = _map_to_ptr[self._outtype](self._ctxt, SA)
    if (Aobj == -1 or SAobj == -1):
      return -1

    _lib.sl_apply_sketch_transform(self._obj, Aobj, SAobj, dim)

    if _map_to_ptr_cleaner.has_key(self._intype):
      _map_to_ptr_cleaner[self._intype](Aobj)

    if _map_to_ptr_cleaner.has_key(self._outtype):
      _map_to_ptr_cleaner[self._outtype](SAobj)

#
# Various sketch transforms
#

class JLT(SketchTransform):
  """
  Johnson-Lindenstrauss Transform
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(JLT, self).__init__(ctxt, "JLT", intype, outtype, n, s);

class CT(SketchTransform):
  """
  Cauchy Transform
  """
  def __init__(self, ctxt, intype, outtype, n, s, C):
    self._obj = _lib.sl_create_sketch_transform(ctxt._obj, "CT", intype, outtype, n, s, ctypes.c_double(C))

class FJLT(SketchTransform):
  """
  Fast Johnson-Lindenstrauss Transform
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(FJLT, self).__init__(ctxt, "FJLT", intype, outtype, n, s);

class CWT(SketchTransform):
  """
  Clarkson-Woodruff Transform (also known as CountSketch)

  *K. Clarkson* and *D. Woodruff*, **Low Rank Approximation and Regression
  in Input Sparsity Time**, STOC 2013 
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(CWT, self).__init__(ctxt, "CWT", intype, outtype, n, s);

class MMT(SketchTransform):
  """
  Meng-Mahoney Transform

  *X. Meng* and *M. W. Mahoney*, **Low-distortion Subspace Embeddings in
  Input-sparsity Time and Applications to Robust Linear Regression**, STOC 2013
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(MMT, self).__init__(ctxt, "MMT", intype, outtype, n, s);

class WZT(SketchTransform):
  """
  Woodruff-Zhang Transform

  *D. Woodruff* and *Q. Zhang*, **Subspace Embeddings and L_p Regression
  Using Exponential Random**, COLT 2013
  """
  def __init__(self, ctxt, intype, outtype, n, s, p):
    self._obj = _lib.sl_create_sketch_transform(ctxt._obj, "WZT", intype, outtype, n, s, ctypes.c_double(p))

class GaussianRFT(SketchTransform):
  """
  Random Features Transform for the RBF Kernel
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(GaussianRFT, self).__init__(ctxt, "GaussianRFT", intype, outtype, n, s);

class LaplacianRFT(SketchTransform):
  """
  Random Features Transform for the Laplacian Kernel

  *A. Rahimi* and *B. Recht*, **Random Features for Large-scale 
  Kernel Machines*, NIPS 2009
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(LaplacianRFT, self).__init__(ctxt, "LaplacianRFT", intype, outtype, n, s);

