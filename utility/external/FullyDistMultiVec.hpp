#ifndef FULLY_DIST_MULTI_VEC_HPP
#define FULLY_DIST_MULTI_VEC_HPP

#include "CombBLAS.h"

template <typename IndexType, typename ValueType>
struct FullyDistMultiVec {
  typedef IndexType index_t; 
  typedef ValueType value_t;
  typedef FullyDistVec<IndexType, ValueType> mpi_vector_t;
  typedef std::vector<mpi_vector_t*> container_t;

  index_t dim;
  index_t size;
  container_t multi_vec_container;

  void clear () { 
    for (index_t i=0; i<size; ++i) 
      if (NULL!=multi_vec_container[i]) 
        delete multi_vec_container[i];
  }

  FullyDistMultiVec (const FullyDistMultiVec& other) :
    dim (other.dim), size(other.size), multi_vec_container (size) {
    for (index_t i=0; i<size; ++i) 
      multi_vec_container[i] = new mpi_vector_t(other[i]);
  }

  /**
   * Construct a multi-vector with N vectors of length M and init-val
   */ 
  FullyDistMultiVec (index_t dim, 
                     index_t size, 
                     value_t init_val=0.0) : 
                     dim(dim), size(size), multi_vec_container(size) {
    for (index_t i=0; i<size; ++i) 
      multi_vec_container[i] = new mpi_vector_t(dim,init_val);
  }

  mpi_vector_t& operator[](index_t i) { return *(multi_vec_container[i]); }

  mpi_vector_t& operator[](index_t i) const {return *(multi_vec_container[i]);}

  void operator=(const FullyDistMultiVec& other) {
    clear();
    dim = other.dim;
    size = other.size;
    multi_vec_container.resize(other.size);
    for (index_t i=0; i<size; ++i) 
      multi_vec_container[i] = new mpi_vector_t(other[i]);
  }
};

#endif // FULLY_DIST_MULTI_VEC_HPP
