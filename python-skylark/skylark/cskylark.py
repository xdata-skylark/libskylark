import ctypes, numpy, sys
from ctypes import byref, cdll, c_double, c_void_p, c_int, pointer, POINTER
import os

lib = cdll.LoadLibrary('libcskylark.so')

#
# Return types
#

lib.sl_create_context.restype = c_void_p
lib.sl_create_default_context.restype = c_void_p
lib.sl_context_rank.restype = c_int
lib.sl_context_size.restype = c_int
lib.sl_create_sketch_transform.restype = c_void_p

class Context(object):
  """
  Create a Skylark Context Object
  """
  def __init__(self, seed):
    self.obj = lib.sl_create_default_context(seed)
    return
  # TODO: Figure out how to wrap MPI_Comm

  def Free(self):
    lib.sl_free_context(self.obj)
    self.obj = 0

  def Size(self):
    return lib.sl_context_size(self.obj)

  def Rank(self):
    return lib.sl_context_rank(self.obj)


class SketchTransform(object):
  """
  Base class sketch transforms.
  The various sketch transforms derive from this class and 
  which holds the common interface. Derived classes can have different constructors.
  """
  def __init__(self, ctxt, ttype, intype, outtype, n, s):
    self.obj = lib.sl_create_sketch_transform(ctxt.obj, ttype, intype, outtype, n, s)
    return

  def Free(self):
    """ Discard the transform """
    lib.sl_free_sketch_transform(self.obj)
    self.obj = 0
    return

  def Apply(self, A, SA, dim):
    """
    Apply the transform on **A** along dimension **dim** and write
    result in **SA**.

    :param A: Input matrix.
    :param SA: Ouptut matrix.
    :param dim: Dimension to apply along. 1 - columnwise, 2 - rowwise.

    """
    #TODO: is there a more elegant way to distinguish matrix types?
    #XXX: e.g. issubclass(type(mat), pyCombBLAS.pySpParMat)
    if str(type(A)).find("elem") is not -1:
        Aobj  = A.obj
        SAobj = SA.obj
    elif str(type(A._m_)).find("pyCombBLAS") is not -1:
        Aobj  = ctypes.c_void_p(long(A._m_.this))
        SAobj = ctypes.c_void_p(long(SA._m_.this))
    else:
        print("unknown type (%s) of matrix in apply()!" % (str(type(A))))
        return -1

    lib.sl_apply_sketch_transform(self.obj, Aobj, SAobj, dim)
    return


class JLT(SketchTransform):
  """
  Johnson-Lindenstrauss Transform
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(JLT, self).__init__(ctxt, "JLT", intype, outtype, n, s);
    return

class CT(SketchTransform):
  """
  Cauchy Transform
  """
  def __init__(self, ctxt, intype, outtype, n, s, C):
    self.obj = lib.sl_create_sketch_transform(ctxt.obj, "CT", intype, outtype, n, s, ctypes.c_double(C))
    return

class FJLT(SketchTransform):
  """
  Fast Johnson-Lindenstrauss Transform
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(FJLT, self).__init__(ctxt, "FJLT", intype, outtype, n, s);
    return

class CWT(SketchTransform):
  """
  Clarkson-Woodruff Transform (also known as CountSketch)

  *K. Clarkson* and *D. Woodruff*, **Low Rank Approximation and Regression in Input Sparsity Time**, STOC 2013 
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(CWT, self).__init__(ctxt, "CWT", intype, outtype, n, s);
    return

class MMT(SketchTransform):
  """
  Meng-Mahoney Transform

  *X. Meng* and *M. W. Mahoney*, **Low-distortion Subspace Embeddings in Input-sparsity Time and Applications to Robust Linear Regression**, STOC 2013
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(MMT, self).__init__(ctxt, "MMT", intype, outtype, n, s);
    return

class WZT(SketchTransform):
  """
  Woodruff-Zhang Transform

  *D. Woodruff* and *Q. Zhang*, **Subspace Embeddings and L_p Regression Using Exponential Random**, COLT 2013
  """
  def __init__(self, ctxt, intype, outtype, n, s, p):
    self.obj = lib.sl_create_sketch_transform(ctxt.obj, "WZT", intype, outtype, n, s, ctypes.c_double(p))
    return

class GaussianRFT(SketchTransform):
  """
  Random Features Transform for the RBF Kernel
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(GaussianRFT, self).__init__(ctxt, "GaussianRFT", intype, outtype, n, s);
    return

class LaplacianRFT(SketchTransform):
  """
  Random Features Transform for the Laplacian Kernel

  *A. Rahimi* and *B. Recht*, **Random Features for Large-scale Kernel Machines*, NIPS 2009
  """
  def __init__(self, ctxt, intype, outtype, n, s):
    super(LaplacianRFT, self).__init__(ctxt, "LaplacianRFT", intype, outtype, n, s);
    return

