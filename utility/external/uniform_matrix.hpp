#ifndef SKYLARK_UNIFORM_MATRIX_HPP
#define SKYLARK_UNIFORM_MATRIX_HPP

#include "../../config.h"
#include "../../sketch/context.hpp"

#if SKYLARK_HAVE_BOOST
#include <boost/random.hpp>
#endif 

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif 

#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif

namespace skylark { namespace utility {

#if SKYLARK_HAVE_BOOST

template <typename ValueType> struct uniform_dist_t {};

template <> struct uniform_dist_t <double> {
  boost::random::mt19937 prng;
  boost::uniform_real<double> distribution;

  uniform_dist_t (int seed) { prng.seed (seed); }

  uniform_dist_t (int seed, double low, double high) : 
      distribution(low, high) { prng.seed(seed); }

  double apply() { return distribution(prng); }
};

template <> struct uniform_dist_t <int> {
  boost::random::mt19937 prng;
  boost::uniform_int<int> distribution;

  uniform_dist_t (int seed) { prng.seed (seed); }

  uniform_dist_t (int seed, int low, int high) : 
      distribution(low, high) { prng.seed(seed); }

  int apply() { return distribution(prng); }
};

template <> struct uniform_dist_t <bool> {
  boost::random::mt19937 prng;
  boost::uniform_int<int> distribution;

  uniform_dist_t (int seed) : distribution(0,1) { prng.seed (seed); }

  bool apply() { return (1==distribution(prng)); }
};

#endif // SKYLARK_HAVE_BOOST

/**
 * A structure to populate the matrix with uniformly dist random entries.
 * If the same seed is used, the same entries are (technically) generated.
 */ 
template <typename MatrixOrVectorType>
struct uniform_matrix_t {};

#if SKYLARK_HAVE_COMBBLAS && SKYLARK_HAVE_BOOST

/**
 * Specialization for a fully distributed dense vector.
 */ 
template <typename IndexType,
          typename ValueType>
struct uniform_matrix_t <FullyDistVec<IndexType, ValueType> > {
  typedef ValueType value_t;
  typedef IndexType index_t;
  typedef FullyDistVec<IndexType,ValueType> mpi_vector_t;
  typedef uniform_dist_t<ValueType> prng_t;

  static mpi_vector_t apply (index_t& M, 
                             skylark::sketch::context_t& context) {
    /* Create the random number generator */
    prng_t my_prng(context.newseed()+context.rank);

    /* Create a dummy vector */
    mpi_vector_t x(M, 0);

    /* Iterate and fill up the local entries */
    for (index_t i=0; i<x.TotalLength(); ++i) x.SetElement(i,my_prng.apply());

    return x;
  }
};

template <typename IndexType,
          typename ValueType>
struct uniform_matrix_t <FullyDistMultiVec<IndexType, ValueType> > {
  typedef ValueType value_t;
  typedef IndexType index_t;
  typedef FullyDistVec<IndexType,ValueType> mpi_vector_t;
  typedef FullyDistMultiVec<IndexType,ValueType> mpi_multi_vector_t;

  static mpi_multi_vector_t apply (index_t M,
                                   index_t N,
                                   skylark::sketch::context_t& context) {
    /* Create an empty multi-vector */
    mpi_multi_vector_t X(M/*dimension*/, N/*number of vectors*/);

    /* Just pass each individual vector to the uniform_matrix_t above */
    for (index_t i=0; i<X.size; ++i)
      X[i] = uniform_matrix_t<mpi_vector_t>::apply(M, context);

    return X;
  }
};

template <typename IndexType,
          typename ValueType>
struct uniform_matrix_t <SpParMat<IndexType, 
                                  ValueType, 
                                  SpDCCols<IndexType, ValueType> > > {
  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef SpDCCols<index_t,value_t> seq_matrix_t;
  typedef SpParMat<index_t,value_t, seq_matrix_t> mpi_matrix_t;
  typedef FullyDistVec<IndexType,ValueType> mpi_value_vector_t;
  typedef FullyDistVec<IndexType,IndexType> mpi_index_vector_t;

  static mpi_matrix_t apply(index_t M, 
                            index_t N, 
                            index_t NNZ,
                            skylark::sketch::context_t& context) {
    /* Create three FullyDistVec for colid, rowid, and values */
    mpi_value_vector_t values = 
      uniform_matrix_t<mpi_value_vector_t>::apply(NNZ, context);
    mpi_index_vector_t col_id(NNZ, 0);
    mpi_index_vector_t row_id(NNZ, 0);

    /* Add edges carefully */
    index_t total_num_edges_added = 0;
    uniform_dist_t<bool> my_prng(context.newseed());
    for (index_t j=0; j<N; ++j) {
      for (index_t i=0; i<M; ++i) {
        if (my_prng.apply()) {
          col_id.SetElement(total_num_edges_added, j);
          row_id.SetElement(total_num_edges_added, i);
          ++total_num_edges_added;
          if (NNZ==total_num_edges_added) break;
        }
      }
      if (NNZ==total_num_edges_added) break;
    }

    return mpi_matrix_t (M, N, row_id, col_id, values);
  }
};

#endif // SKYLARK_HAVE_COMBBLAS and SKYLARK_HAVE_BOOST

#if SKYLARK_HAVE_ELEMENTAL && SKYLARK_HAVE_BOOST

template <typename ValueType>
struct uniform_matrix_t <elem::Matrix<ValueType> > {
  typedef int index_t;
  typedef ValueType value_t;
  typedef elem::Matrix<ValueType> matrix_t;

  static matrix_t apply (index_t M,
                         index_t N,
                         skylark::sketch::context_t& context) {
    matrix_t A(M, N);
    elem::MakeUniform(A);
    return A;
  }
};

template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RD>
struct uniform_matrix_t <elem::DistMatrix<ValueType, CD, RD> > {
  typedef int index_t;
  typedef ValueType value_t;
  typedef elem::DistMatrix<ValueType, CD, RD> mpi_matrix_t;

  static mpi_matrix_t apply (index_t M,
                             index_t N,
                             elem::Grid& grid,
                             skylark::sketch::context_t& context) {
    mpi_matrix_t A(M, N, grid);
    elem::MakeUniform (A);
    return A;
  }
};

#endif /** SKYLARK_HAVE_ELEMENTAL and SKYLARK_HAVE_BOOST */

} } /** namespace skylark::utlity */

#endif // SKYLARK_UNIFORM_MATRIX_HPP
