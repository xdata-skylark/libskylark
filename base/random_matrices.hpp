#ifndef SKYLARK_RANDOM_MATRICES_HPP
#define SKYLARK_RANDOM_MATRICES_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

#include "../utility/typer.hpp"

namespace skylark { namespace base {

/**
 * Generate random matrix using specificed distribution
 * (i.i.d samples).
 *
 * Implementation for local matrices.
 *
 * \param A Output matrix
 * \param m,n Number of rows and colunt
 * \param dist Distribution object
 * \param context Skylark context.
 */
template<typename T, template<typename> class DistributionType>
void RandomMatrix(El::Matrix<T> &A, El::Int m, El::Int n,
    DistributionType<T> &dist, context_t &context) {

    random_samples_array_t< DistributionType<T> > entries =
        context.allocate_random_samples_array(m * n, dist);

    A.Resize(m, n);
    T *data = A.Buffer();

#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(size_t j = 0; j < n; j++)
        for(size_t i = 0; i < m; i++)
            data[j * m + i] = entries[j * m + i];
}

template<typename T, template<typename, typename> class DistributionType>
void RandomMatrix(El::Matrix<T> &A, El::Int m, El::Int n,
    DistributionType<T, T> &dist, context_t &context) {

    random_samples_array_t< DistributionType<T, T> > entries =
        context.allocate_random_samples_array(m * n, dist);

    A.Resize(m, n);
    T *data = A.Buffer();

#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(size_t j = 0; j < n; j++)
        for(size_t i = 0; i < m; i++)
            data[j * m + i] = entries[j * m + i];
}

/**
 * Generate random matrix using specificed distribution
 * (i.i.d samples).
 *
 * Implementation for distributed matrices.
 *
 * \param A Output matrix
 * \param m,n Number of rows and colunt
 * \param dist Distribution object
 * \param context Skylark context.
 */
template<typename T, El::Distribution CD, El::Distribution RD,
         template<typename> class DistributionType>
void RandomMatrix(El::DistMatrix<T, CD, RD> &A, El::Int m, El::Int n,
    DistributionType<T> &dist, context_t &context) {

    random_samples_array_t< DistributionType<T> > entries =
        context.allocate_random_samples_array(m * n, dist);

    A.Resize(m, n);

    size_t m0 = A.LocalHeight();
    size_t n0 = A.LocalWidth();
    T *data = A.Buffer();

#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(size_t j = 0; j < n0; j++)
        for(size_t i = 0; i < m0; i++)
            data[j * m0 + i] = entries[A.GlobalCol(j) * m + A.GlobalRow(i)];
}

/**
 * Generate random matrix using specificed distribution
 * (i.i.d samples).
 *
 * Implementation for distributed matrices.
 *
 * \param A Output matrix
 * \param m,n Number of rows and colunt
 * \param dist Distribution object
 * \param context Skylark context.
 */
template<typename T, El::Distribution CD, El::Distribution RD,
         template<typename, typename> class DistributionType>
void RandomMatrix(El::DistMatrix<T, CD, RD> &A, El::Int m, El::Int n,
    DistributionType<T, T> &dist, context_t &context) {

    random_samples_array_t< DistributionType<T, T> > entries =
        context.allocate_random_samples_array(m * n, dist);

    A.Resize(m, n);

    size_t m0 = A.LocalHeight();
    size_t n0 = A.LocalWidth();
    T *data = A.Buffer();

#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(size_t j = 0; j < n0; j++)
        for(size_t i = 0; i < m0; i++)
            data[j * m0 + i] = entries[A.GlobalCol(j) * m + A.GlobalRow(i)];
}

/**
 * Generate random matrix with i.i.d standard Gaussian entries.
 *
 * \param A Output matrix.
 * \param m,n Number of rows and columns.
 * \param context Skylark context.
 */
template<typename MatrixType>
void GaussianMatrix(MatrixType &A, El::Int m, El::Int n,
    context_t &context) {
    typedef typename utility::typer_t<MatrixType>::value_type value_type;

    boost::random::normal_distribution<value_type> dist;
    RandomMatrix(A, m, n, dist, context);
}

/**
 * Generate random matrix with i.i.d [0,1) uniform entries.
 *
 * \param A Output matrix.
 * \param m,n Number of rows and columns.
 * \param context Skylark context.
 */
template<typename MatrixType>
void UniformMatrix(MatrixType &A, El::Int m, El::Int n,
    context_t &context) {
    typedef typename utility::typer_t<MatrixType>::value_type value_type;

    boost::random::uniform_01<value_type> dist;
    RandomMatrix(A, m, n, dist, context);
}

} } // namespace skylark::base

#endif // SKYLARK_RANDOM_MATRICES_HPP
