from cython.operator cimport dereference as deref


cdef extern from "<limits>" namespace "std":
    cdef size_t max_index "std::numeric_limits<size_t>::max"()


cdef extern from "boost/random.hpp" namespace "boost::random":
     cdef cppclass normal_distribution[T]:
         normal_distribution(T, T)
         T get "operator()"[Engine](Engine&)

     cdef cppclass uniform_int_distribution[T]:
         uniform_int_distribution(T, T)
         T get "operator()"[Engine](Engine&)

     cdef cppclass uniform_real_distribution[T]:
         uniform_real_distribution(T, T)
         T get "operator()"[Engine](Engine&)

     cdef cppclass cauchy_distribution[T]:
         cauchy_distribution(T, T)
         T get "operator()"[Engine](Engine&)

     cdef cppclass exponential_distribution[T]:
         exponential_distribution(T)
         T get "operator()"[Engine](Engine&)


cdef extern from "skylark/utility/exception.hpp" namespace "skylark::utility":
     pass


cdef extern from "skylark/utility/randgen.hpp" namespace "skylark::utility":
     cdef cppclass random_samples_array_t[Distribution]:
         random_samples_array_t(size_t base, size_t size, int seed, Distribution& distribution)   
         double at "operator[]"(size_t index) except +


cdef extern from "skylark/utility/distributions.hpp" namespace "skylark::utility":
     cdef cppclass standard_levy_distribution_t[T]:
         standard_levy_distribution_t()
         T get "operator()"[Engine](Engine&)

     cdef cppclass rademacher_distribution_t[T]:
         rademacher_distribution_t()
         T get "operator()"[Engine](Engine&)


