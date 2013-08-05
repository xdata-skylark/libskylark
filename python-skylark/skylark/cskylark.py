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

# 
# Create mapping between object type to function converting it 
# to pointers to passed to the C-API
#
def _elem_to_ptr(A):
  return ctypes.c_void_p(long(A.this));

def _kdt_to_ptr(A):
    return ctypes.c_void_p(long(A._m_.this));

_map_to_ptr = { }
_map_to_ptr["DistMatrix_d_VR_STAR"] = _elem_to_ptr
_map_to_ptr["DistMatrix_d_VC_STAR"] = _elem_to_ptr
_map_to_ptr["DistMatrix_d_STAR_VR"] = _elem_to_ptr
_map_to_ptr["DistMatrix_d_STAR_VC"] = _elem_to_ptr
_map_to_ptr["instance"] = _kdt_to_ptr

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
    self._obj = _lib.sl_create_sketch_transform(ctxt._obj, ttype, intype, outtype, n, s)

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
    if not _map_to_ptr.has_key(type(A).__name__):
      print "unknown type (%s) of A in apply()!" % type(A).__name__
      return -1

    if not _map_to_ptr.has_key(type(SA).__name__):
      print "unknown type (%s) of SA in apply()!" % type(SA).__name__
      return -1

    Aobj = _map_to_ptr[type(A).__name__](A);
    SAobj = _map_to_ptr[type(SA).__name__](SA);

    _lib.sl_apply_sketch_transform(self._obj, Aobj, SAobj, dim)

    #TODO: is there a more elegant way to distinguish matrix types?
    #XXX: e.g. issubclass(type(mat), pyCombBLAS.pySpParMat)
    #if str(type(A)).find("elem") is not -1:
    #    Aobj  = ctypes.c_void_p(long(A.this))
    #    SAobj = ctypes.c_void_p(long(SA.this))
    #elif str(type(A._m_)).find("pyCombBLAS") is not -1:
    #    Aobj  = ctypes.c_void_p(long(A._m_.this))
    #    SAobj = ctypes.c_void_p(long(SA._m_.this))
    #else:
    #    print("unknown type (%s) of matrix in apply()!" % (str(type(A))))
    #    return -1

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

  *K. Clarkson* and *D. Woodruff*, **Low Rank Approximation and Regression in Input Sparsity Time**, STOC 2013 
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(CWT, self).__init__(ctxt, "CWT", intype, outtype, n, s);

class MMT(SketchTransform):
  """
  Meng-Mahoney Transform

  *X. Meng* and *M. W. Mahoney*, **Low-distortion Subspace Embeddings in Input-sparsity Time and Applications to Robust Linear Regression**, STOC 2013
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(MMT, self).__init__(ctxt, "MMT", intype, outtype, n, s);

class WZT(SketchTransform):
  """
  Woodruff-Zhang Transform

  *D. Woodruff* and *Q. Zhang*, **Subspace Embeddings and L_p Regression Using Exponential Random**, COLT 2013
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

  *A. Rahimi* and *B. Recht*, **Random Features for Large-scale Kernel Machines*, NIPS 2009
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(LaplacianRFT, self).__init__(ctxt, "LaplacianRFT", intype, outtype, n, s);

