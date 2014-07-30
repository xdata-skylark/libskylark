#ifndef SKYLARK_EMPTY_MATRIX_HPP
#define SKYLARK_EMPTY_MATRIX_HPP

#include "config.h"

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif

#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif

namespace skylark { namespace utility {

/**
 * A structure to create an empty matrix.
 */
template <typename MatrixOrVectorType>
struct empty_matrix_t {};

#if SKYLARK_HAVE_COMBBLAS

/**
 * Specialization for a fully distributed dense vector.
 */
template <typename IndexType,
          typename ValueType>
struct empty_matrix_t <FullyDistVec<IndexType, ValueType> > {
  typedef ValueType value_t;
  typedef IndexType index_t;
  typedef FullyDistVec<IndexType,ValueType> mpi_vector_t;

  static mpi_vector_t generate (index_t& M) {
    return mpi_vector_t(M, 0);
  }
};

template <typename IndexType,
          typename ValueType>
struct empty_matrix_t <FullyDistMultiVec<IndexType, ValueType> > {
  typedef ValueType value_t;
  typedef IndexType index_t;
  typedef FullyDistVec<IndexType,ValueType> mpi_vector_t;
  typedef FullyDistMultiVec<IndexType,ValueType> mpi_multi_vector_t;

  static mpi_multi_vector_t generate (index_t M,
                                   index_t N) {
    /* Create an empty multi-vector */
    return mpi_multi_vector_t(M /* dimension */,
                              N /* number of vectors */,
                              0 /* intial value */);
  }
};

template <typename IndexType,
          typename ValueType>
struct empty_matrix_t <SpParMat<IndexType,
                                ValueType,
                                SpDCCols<IndexType, ValueType> > > {
  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef SpDCCols<index_t,value_t> seq_matrix_t;
  typedef SpParMat<index_t,value_t, seq_matrix_t> mpi_matrix_t;
  typedef FullyDistVec<IndexType,ValueType> mpi_value_vector_t;
  typedef FullyDistVec<IndexType,IndexType> mpi_index_vector_t;

  static mpi_matrix_t generate (index_t M,
                            index_t N) {
    return mpi_matrix_t (M,
                         N,
                         mpi_index_vector_t(0, 0),
                         mpi_index_vector_t(0, 0),
                         mpi_index_vector_t(0, 0));
  }
};

#endif // SKYLARK_HAVE_COMBBLAS

#if SKYLARK_HAVE_ELEMENTAL

template <typename ValueType>
struct empty_matrix_t <elem::Matrix<ValueType> > {
  typedef int index_t;
  typedef ValueType value_t;
  typedef elem::Matrix<ValueType> matrix_t;

  static matrix_t generate (index_t M,
                         index_t N) {
    return matrix_t (M, N);
  }
};

template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RD>
struct empty_matrix_t <elem::DistMatrix<ValueType, CD, RD> > {
  typedef int index_t;
  typedef ValueType value_t;
  typedef elem::DistMatrix<ValueType, CD, RD> mpi_matrix_t;

  static mpi_matrix_t generate (index_t M,
                             index_t N,
                             elem::Grid& grid) {
    return mpi_matrix_t (M, N, grid);
  }
};

#endif // SKYLARK_HAVE_ELEMENTAL

} } /** namespace skylark::utlity */

#endif // SKYLARK_EMPTY_MATRIX_HPP
