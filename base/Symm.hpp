#ifndef SKYLARK_SYMM_HPP
#define SKYLARK_SYMM_HPP

#include <boost/mpi.hpp>
#include "exception.hpp"
#include "sparse_matrix.hpp"
#include "computed_matrix.hpp"
#include "../utility/typer.hpp"


// Defines a generic Symm function that receives both dense and sparse matrices.

namespace skylark { namespace base {

/**
 * Rename the Elemental Symm function, so that we have unified access.
 */

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const El::Matrix<T>& A, const El::Matrix<T>& B,
    T beta, El::Matrix<T>& C) {
    El::Symm(side, uplo, alpha, A, B, beta, C);
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const El::Matrix<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& C) {
    El::Symm(side, uplo, alpha, A, B, C);
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const El::DistMatrix<T, El::STAR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    T beta, El::DistMatrix<T, El::STAR, El::STAR>& C) {
    El::Symm(side, uplo,  alpha, A.LockedMatrix(), B.LockedMatrix(), 
        beta, C.Matrix());
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const El::DistMatrix<T, El::STAR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& C) {
    El::Symm(side, uplo, alpha, A.LockedMatrix(), B.LockedMatrix(), C.Matrix());
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const El::DistMatrix<T>& A, const El::DistMatrix<T>& B,
    T beta, El::DistMatrix<T>& C) {
    El::Symm(side, uplo, alpha, A, B, beta, C);
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const El::DistMatrix<T>& A, const El::DistMatrix<T>& B,
    El::DistMatrix<T>& C) {
    El::Symm(side, uplo, alpha, A, B, T(0.0), C);
}

} } // namespace skylark::base

#endif // SKYLARK_SYMM_HPP
