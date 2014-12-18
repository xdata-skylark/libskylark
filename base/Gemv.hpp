#ifndef SKYLARK_GEMV_HPP
#define SKYLARK_GEMV_HPP

#include <boost/mpi.hpp>
#include "exception.hpp"

// Defines a generic Gemv function that recieves a wider set of matrices

namespace skylark { namespace base {

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::Matrix<T>& A, const El::Matrix<T>& x,
    T beta, El::Matrix<T>& y) {
    El::Gemv(oA, alpha, A, x, beta, y);
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::Matrix<T>& A, const El::Matrix<T>& x,
    El::Matrix<T>& y) {
    El::Gemv(oA, alpha, A, x, y);
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::DistMatrix<T>& A, const El::DistMatrix<T>& x,
    T beta, El::DistMatrix<T>& y) {
    El::Gemv(oA, alpha, A, x, beta, y);
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::DistMatrix<T>& A, const El::DistMatrix<T>& x,
    El::DistMatrix<T>& y) {
    El::Gemv(oA, alpha, A, x, y);
}

/**
 * The following combinations is not offered by Elental, but is useful for us.
 * We implement it partially.
 */

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& x,
    T beta, El::DistMatrix<T, El::STAR, El::STAR>& y) {
    // TODO verify sizes etc.
    // TODO verify matching grids.

    if (oA == El::TRANSPOSE) {
        boost::mpi::communicator comm(y.Grid().Comm().comm,
            boost::mpi::comm_attach);
        El::Matrix<T> ylocal(y.Matrix());
        El::Gemv(El::TRANSPOSE,
            alpha, A.LockedMatrix(), x.LockedMatrix(),
            beta / T(comm.size()), ylocal);
        boost::mpi::all_reduce(comm,
            ylocal.Buffer(), ylocal.MemorySize(), y.Buffer(),
            std::plus<T>());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& x,
    El::DistMatrix<T, El::STAR, El::STAR>& y) {

    int y_height = (oA == El::NORMAL ? A.Height() : A.Width());
    El::Zeros(y, y_height, 1);
    base::Gemv(oA, alpha, A, x, T(0), y);
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& x,
    T beta, El::DistMatrix<T, El::VC, El::STAR>& y) {
    // TODO verify sizes etc.

    if (oA == El::NORMAL) {
        El::Gemv(El::NORMAL,
            alpha, A.LockedMatrix(), x.LockedMatrix(),
            beta, y.Matrix());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& x,
    El::DistMatrix<T, El::VC, El::STAR>& y) {

    int y_height = (oA == El::NORMAL ? A.Height() : A.Width());
    El::Zeros(y, y_height, 1);
    base::Gemv(oA, alpha, A, x, T(0), y);
}

template<typename T>
inline void Gemv(El::Orientation oA,
    T alpha, const sparse_matrix_t<T>& A, const El::Matrix<T>& x,
    T beta, El::Matrix<T>& y) {
    // TODO verify sizes etc.

    const int* indptr = A.indptr();
    const int* indices = A.indices();
    const double *values = A.locked_values();
    double *yd = y.Buffer();
    const double *xd = x.LockedBuffer();

    int n = A.width();

    if (oA == El::NORMAL) {
        El::Scale(beta, y);

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int col = 0; col < n; col++) {
            T xv = alpha * xd[col];
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                     int row = indices[j];
                     T val = values[j];
                     yd[row] += val * xv;
                 }
        }

    } else {

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int col = 0; col < n; col++) {
            double yv = beta * yd[col];
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                     int row = indices[j];
                     T val = values[j];
                     yv += alpha * val * xd[row];
                 }
            yd[col] = yv;
        }

    }
}

} } // namespace skylark::base


#endif // SKYLARK_GEMV_HPP
