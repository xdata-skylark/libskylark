import El
import skylark.io as sl_io
import skylark.ml.kernels as sl_kernels
import skylark.ml.rlsc as sl_rlsc

from ctypes import byref, c_void_p, c_double, c_int

class Classifier(object):
  
  def load_data(self, fname, dir="columns", min_d=0, left_type=El.dTag, right_type=El.iTag):
    self.left_type = left_type
    self.right_type = right_type
    self.dir = dir
    self.min_d = min_d


    self.A = El.DistMatrix(left_type)
    self.b = El.DistMatrix(right_type)
    sl_io.readlibsvm(fname, self.A, self.b, dir, min_d=min_d)


  def load_test_data(self, fname):
    self.Atest = El.DistMatrix(self.left_type)
    self.btest = El.DistMatrix(self.right_type)
    sl_io.readlibsvm(fname, self.Atest, self.btest, self.dir, min_d=self.min_d)
    El.Transpose(self.btest, self.btest)


  def set_kernel(self, kernel, *args):
    if isinstance(kernel, sl_kernels.Kernel):
      self.kernel = kernel
    elif isinstance(kernel, str):
      raise ValueError("creating kernel from a classifier not implemented yet")
    else:
      raise ValueError("kernel not recognized")


  def train(self, algorithm="krr", *args):
    self.x = El.DistMatrix()

    if algorithm == "krr":
      (self.x, self.rcoding) = sl_rlsc.kernel_rlsc(self.A, self.b, self.x, \
        self.kernel, args[0], dir=self.dir)
    elif algorithm == "approximate_krr":
      raise ValueError("Approximate kernel ridge is not implemented yet")
  

  def predict(self):
    KT = El.DistMatrix()
    self.kernel.gram(self.A, KT, self.dir, self.dir, self.Atest)

    self.DV = El.DistMatrix()
    self.DV.Resize(self.x.Width(), KT.Width())

    El.Gemm(El.ADJOINT, El.NORMAL, 1.0, self.x, KT, 0.0, self.DV)

    self.predictions = self.dummyDecode()
    return self.predictions


  def error_rate(self):      
    errors = 0.0

    n = self.btest.Height()
    for i in xrange(n):
      if self.btest.Get(i, 0) != self.predictions.Get(i, 0):
        errors += 1

    return errors/n

  def dummyDecode(self):
    predictions = El.DistMatrix()
    predictions.Resize(self.DV.Width(), 1)
    
    for i in xrange(self.DV.Width()):
      pos = 0
      for j in xrange(self.DV.Height()):
        if self.DV.Get(j, i) > self.DV.Get(pos, i):
          pos = j
      predictions.Set(i, 0, self.rcoding.Get(0, pos))
    
    self.predictions = predictions
    return self.predictions
