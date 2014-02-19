
from cython.operator cimport dereference as deref
from utility cimport *


cdef class normal_distribution_array:
   cdef random_samples_array_t[normal_distribution[double] ] *thisptr
   cdef normal_distribution[double] *distribution
   
   def __cinit__(self, size_t base, int seed, mean_arg = 0.0, sigma_arg = 1.0):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new normal_distribution[double](mean_arg, sigma_arg)
       self.thisptr  = new random_samples_array_t[normal_distribution[double] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <double>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


cdef class uniform_int_distribution_array:
   cdef random_samples_array_t[uniform_int_distribution[int] ] *thisptr
   cdef uniform_int_distribution[int] *distribution
   
   def __cinit__(self, size_t base, int seed, min_arg = 0, max_arg = 10):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new uniform_int_distribution[int](min_arg, max_arg)
       self.thisptr  = new random_samples_array_t[uniform_int_distribution[int] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <int>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


cdef class uniform_real_distribution_array:
   cdef random_samples_array_t[uniform_real_distribution[double] ] *thisptr
   cdef uniform_real_distribution[double] *distribution
   
   def __cinit__(self, size_t base, int seed, min_arg = 0.0, max_arg = 1.0):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new uniform_real_distribution[double](min_arg, max_arg)
       self.thisptr  = new random_samples_array_t[uniform_real_distribution[double] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <double>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


cdef class cauchy_distribution_array:
   cdef random_samples_array_t[cauchy_distribution[double] ] *thisptr
   cdef cauchy_distribution[double] *distribution
   
   def __cinit__(self, size_t base, int seed, median_arg = 0.0, sigma_arg = 1.0):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new cauchy_distribution[double](median_arg, sigma_arg)
       self.thisptr  = new random_samples_array_t[cauchy_distribution[double] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <double>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


cdef class exponential_distribution_array:
   cdef random_samples_array_t[exponential_distribution[double] ] *thisptr
   cdef exponential_distribution[double] *distribution
   
   def __cinit__(self, size_t base, int seed, lambda_arg = 1.0):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new exponential_distribution[double](lambda_arg)
       self.thisptr  = new random_samples_array_t[exponential_distribution[double] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <double>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


cdef class standard_levy_distribution_t_array:
   cdef random_samples_array_t[standard_levy_distribution_t[double] ] *thisptr
   cdef standard_levy_distribution_t[double] *distribution
   
   def __cinit__(self, size_t base, int seed, ):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new standard_levy_distribution_t[double]()
       self.thisptr  = new random_samples_array_t[standard_levy_distribution_t[double] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <double>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


cdef class rademacher_distribution_t_array:
   cdef random_samples_array_t[rademacher_distribution_t[int] ] *thisptr
   cdef rademacher_distribution_t[int] *distribution
   
   def __cinit__(self, size_t base, int seed, ):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new rademacher_distribution_t[int]()
       self.thisptr  = new random_samples_array_t[rademacher_distribution_t[int] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return <int>self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution


