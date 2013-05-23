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
  def __init__(self, ctxt, ttype, intype, outtype, n, s):
    self.obj = lib.sl_create_sketch_transform(ctxt.obj, ttype, intype, outtype, n, s)
    return
 
  def Free(self):
    lib.sl_free_sketch_transform(self.obj)
    self.obj = 0
    return
 
  def Apply(self, A, SA, dim):
    lib.sl_apply_sketch_transform(self.obj, A.obj, SA.obj, dim)
    return

class JLT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s):
    super(JLT, self).__init__(ctxt, "JLT", intype, outtype, n, s);
    return 

class CT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s, C):
    self.obj = lib.sl_create_sketch_transform(ctxt.obj, "CT", intype, outtype, n, s, ctypes.c_double(C))
    return

class FJLT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s):
    super(FJLT, self).__init__(ctxt, "FJLT", intype, outtype, n, s);
    return 

class CWT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s):
    super(CWT, self).__init__(ctxt, "CWT", intype, outtype, n, s);
    return 

class MMT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s):
    super(MMT, self).__init__(ctxt, "MMT", intype, outtype, n, s);
    return 

class WZT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s, p):
    self.obj = lib.sl_create_sketch_transform(ctxt.obj, "WZT", intype, outtype, n, s, ctypes.c_double(p))
    return 

class GaussianRFT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s):
    super(GaussianRFT, self).__init__(ctxt, "GaussianRFT", intype, outtype, n, s);
    return 

class LaplacianRFT(SketchTransform):
  def __init__(self, ctxt, intype, outtype, n, s):
    super(LaplacianRFT, self).__init__(ctxt, "LaplacianRFT", intype, outtype, n, s);
    return


