import El
import skylark.io as sl_io
import skylark.ml.kernels as sl_kernels
import skylark.ml.rlsc as sl_rlsc
import skylark.lib as lib


from ctypes import byref, c_void_p, c_double, c_int

class Classifier(object):

  def train_krr(self, lambda_value):
    (self.x, self.rcoding) = sl_rlsc.kernel_rlsc(self.A, self.b, self.x, \
        self.kernel, lambda_value, dir=self.dir)

  def predict_krr(self):
    KT = El.DistMatrix()
    self.kernel.gram(self.A, KT, self.dir, self.dir, self.A_test)
    self.DV = El.DistMatrix()
    self.DV.Resize(self.x.Width(), KT.Width())

    El.Gemm(El.ADJOINT, El.NORMAL, 1.0, self.x, KT, 0.0, self.DV)


  def train_approximate_krr(self, lambda_value, s):
    (self.x, self.rcoding, self.S) = sl_rlsc.approximate_kernel_rlsc(self.A, self.b, \
        self.x, self.kernel, lambda_value, s, dir=self.dir)
     
    SA = El.DistMatrix()
    SA.Resize(s, self.A_test.Width())

    Aux = lib.adapt(self.A_test)
    SAux = lib.adapt(SA)

    lib.callsl("sl_apply_sketch_transform_container", self.S, \
      Aux.ptr(), SAux.ptr())
    self.SA = SAux.getobj()
    
  def predict_approximate_krr(self):
    self.DV = El.DistMatrix()
    self.DV.Resize(self.x.Width(), self.SA.Width())

    El.Gemm(El.ADJOINT, El.NORMAL, 1.0, self.x, self.SA, 0.0, self.DV)


  def set_training_data(self, A, b, dir="columns"):
    self.A = A
    self.b = b
    self.dir = dir


  def set_test_data(self, A_test, b_test):
    self.A_test = A_test
    self.b_test = El.DistMatrix(El.iTag)

    if b_test.Height == 1:
      self.b_test = b_test
    else:
      El.Transpose(b_test, self.b_test)


  def set_kernel(self, kernel, *args):
    if isinstance(kernel, sl_kernels.Kernel):
      self.kernel = kernel
    elif isinstance(kernel, str):
      raise ValueError("creating kernel from a classifier not implemented yet")
    else:
      raise ValueError("kernel not recognized")


  def train(self, algorithm="krr", *args):
    self.algorithm = algorithm
    
    self.x = El.DistMatrix()
    
    if algorithm == "krr":
      self.train_krr(args[0])
    elif algorithm == "approximate_krr":
      self.train_approximate_krr(args[0], args[1])
    else:
      raise ValueError(algorithm + " is not implemented")
  


  def predict(self):
    if self.algorithm == "krr":
      self.predict_krr()
    elif self.algorithm == "approximate_krr":
      self.predict_approximate_krr()

    self.predictions = self.dummyDecode()    
    return self.predictions


  def error_rate(self):      
    errors = 0.0
    n = self.b_test.Height()

    for i in xrange(n):
      if self.b_test.Get(i, 0) != self.predictions.Get(i, 0):
        errors += 1

    return errors/n

  def accuracy(self):      
    return 1 - self.error_rate()

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
