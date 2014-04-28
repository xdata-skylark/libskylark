#ifndef SKYLARK_GEMV_HPP
#define SKYLARK_GEMV_HPP

#include <boost/mpi.hpp>
#include "exception.hpp"

// Defines a generic Gemv function that recieves a wider set of matrices

#if SKYLARK_HAVE_ELEMENTAL

namespace skylark { namespace base {

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::Matrix<T>& A, const elem::Matrix<T>& x,
    T beta, elem::Matrix<T>& y) {
    elem::Gemv(oA, alpha, A, x, beta, y);
}

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::Matrix<T>& A, const elem::Matrix<T>& x,
    elem::Matrix<T>& y) {
    elem::Gemv(oA, alpha, A, x, y);
}

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::DistMatrix<T>& A, const elem::DistMatrix<T>& x,
    T beta, elem::DistMatrix<T>& y) {
    elem::Gemv(oA, alpha, A, x, beta, y);
}

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::DistMatrix<T>& A, const elem::DistMatrix<T>& x,
    elem::DistMatrix<T>& y) {
    elem::Gemv(oA, alpha, A, x, y);
}

/**
 * The following combinations is not offered by Elemental, but is useful for us.
 * We implement it partially.
 */

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::DistMatrix<T, elem::VC, elem::STAR>& A,
    const elem::DistMatrix<T, elem::VC, elem::STAR>& x,
    T beta, elem::DistMatrix<T, elem::STAR, elem::STAR>& y) {
    // TODO verify sizes etc.
    // TODO verify matching grids.

    if (oA == elem::TRANSPOSE) {
        boost::mpi::communicator comm(y.Grid().Comm(), boost::mpi::comm_attach);
        elem::Matrix<T> ylocal(y.Matrix());
        elem::Gemv(elem::TRANSPOSE,
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
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::DistMatrix<T, elem::VC, elem::STAR>& A,
    const elem::DistMatrix<T, elem::VC, elem::STAR>& x,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& y) {

    int y_height = (oA == elem::NORMAL ? A.Height() : A.Width());
    elem::Zeros(y, y_height, 1);
    base::Gemv(oA, alpha, A, x, T(0), y);
}

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::DistMatrix<T, elem::VC, elem::STAR>& A,
    const elem::DistMatrix<T, elem::STAR, elem::STAR>& x,
    T beta, elem::DistMatrix<T, elem::VC, elem::STAR>& y) {
    // TODO verify sizes etc.

    if (oA == elem::NORMAL) {
        elem::Gemv(elem::NORMAL,
            alpha, A.LockedMatrix(), x.LockedMatrix(),
            beta, y.Matrix());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemv(elem::Orientation oA,
    T alpha, const elem::DistMatrix<T, elem::VC, elem::STAR>& A,
    const elem::DistMatrix<T, elem::STAR, elem::STAR>& x,
    elem::DistMatrix<T, elem::VC, elem::STAR>& y) {

    int y_height = (oA == elem::NORMAL ? A.Height() : A.Width());
    elem::Zeros(y, y_height, 1);
    base::Gemv(oA, alpha, A, x, T(0), y);
}

} } // namespace skylark::base

#endif // SKYLARK_HAVE_ELEMENTAL

#endif // SKYLARK_GEMV_HPP
