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

// A lazily computed array of uniformly distributed samples
template <typename ValueType> struct uniform_samples_array_t {};

// Specialized for uniform_real_distribution<double>
template <> struct uniform_samples_array_t <double> {
    typedef boost::random::uniform_real_distribution<double> distribution_t;

private:
    skylark::sketch::context_t& _context;
    distribution_t _distribution;
    skylark::utility::random_samples_array_t<double, distribution_t>
    _samples;

public:
    uniform_samples_array_t(int size, skylark::sketch::context_t& context)
        : _context(context), _distribution(distribution_t()),
          _samples(context.allocate_random_samples_array
              <double,distribution_t> (size, _distribution)) {}

    uniform_samples_array_t(int size, double low, double high,
        skylark::sketch::context_t& context)
        : _context(context), _distribution(distribution_t(low, high)),
          _samples(context.allocate_random_samples_array
              <double,distribution_t> (size, _distribution)) {}

    double operator[](int index) {
        return _samples[index];
    }
};

// Specialized for uniform_int_distribution<int>
template <> struct uniform_samples_array_t <int> {
    typedef boost::random::uniform_int_distribution<int> distribution_t;

private:
    skylark::sketch::context_t& _context;
    distribution_t _distribution;
    skylark::utility::random_samples_array_t<int, distribution_t>
    _samples;

public:
    uniform_samples_array_t(int size, double low, double high,
        skylark::sketch::context_t& context)
        : _context(context), _distribution(distribution_t(low, high)),
          _samples(context.allocate_random_samples_array
              <int,distribution_t> (size, _distribution)) {}

    double operator[](int index) {
        return _samples[index];
    }
};

// Specialized for uniform generation of booleans
template <> struct uniform_samples_array_t <bool> {
    typedef boost::random::uniform_int_distribution<int> distribution_t;

private:
    skylark::sketch::context_t& _context;
    distribution_t _distribution;
    skylark::utility::random_samples_array_t<int, distribution_t>
    _samples;

public:
    uniform_samples_array_t(int size, skylark::sketch::context_t& context)
        : _context(context), _distribution(distribution_t(0, 1)),
          _samples(context.allocate_random_samples_array
              <int,distribution_t> (size, _distribution)) {}

    double operator[](int index) {
        return (1 == _samples[index]);
    }
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

  static mpi_vector_t apply (index_t& M,
                             skylark::sketch::context_t& context) {

    /* Create a dummy vector */
    mpi_vector_t x(M, 0);

    uniform_samples_array_t<ValueType> samples(x.TotalLength(), context);

    /* Iterate and fill up the local entries */
    for (index_t i=0; i<x.TotalLength(); ++i) x.SetElement(i, samples[i]);

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

    uniform_samples_array_t<bool> samples(M * N, context);
    for (index_t j=0; j<N; ++j) {
      for (index_t i=0; i<M; ++i) {
        if (samples[j * M + i]) {
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
