from cython.operator cimport dereference as deref


cdef extern from "<limits>" namespace "std":
    cdef size_t max_index "std::numeric_limits<size_t>::max"()


cdef extern from "boost/random.hpp" namespace "boost::random":
     cdef cppclass normal_distribution[T]:
         normal_distribution()
         T get "operator()"[Engine](Engine&)


cdef extern from "skylark/utility/exception.hpp" namespace "skylark::utility":
     pass


cdef extern from "skylark/utility/randgen.hpp" namespace "skylark::utility":
     cdef cppclass random_samples_array_t[Distribution]:
         random_samples_array_t(size_t base, size_t size, int seed, Distribution& distribution)   
         double at "operator[]"(size_t index) except +


cdef class normal_distribution_array:
   cdef random_samples_array_t[normal_distribution[double] ] *thisptr
   cdef normal_distribution[double] *distribution
   
   def __cinit__(self, size_t base, int seed):
       cdef size_t size = max_index() - base + 1
       self.distribution =  new normal_distribution[double]()
       self.thisptr  = new random_samples_array_t[normal_distribution[double] ](base, size, seed, deref(self.distribution))          

   def __getitem__(self, size_t index):
       return self.thisptr.at(index)
       
   def __dealloc__(self):
       del self.thisptr
       del self.distribution
