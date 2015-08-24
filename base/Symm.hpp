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

/**
 * Symm between mixed Elemental, sparse input. Output is dense Elemental.
 */
template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const sparse_matrix_t<T>& A, const El::Matrix<T>& B,
    T beta, El::Matrix<T>& C) {
    // TODO verify sizes etc.

    const int* indptr = A.indptr();
    const int* indices = A.indices();
    const T *values = A.locked_values();

    int k = A.width();
    int n = B.Width();
    int m = B.Height();

    // LEFT
    if (side == El::LEFT) {

        El::Scale(beta, C);

        T *c = C.Buffer();
        int ldc = C.LDim();

        const T *b = B.LockedBuffer();
        int ldb = B.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int i = 0; i < n; i++)
            for(int col = 0; col < k; col++)
                 for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                     int row = indices[j];

                     if ((uplo == El::UPPER && row > col) ||
                         (uplo == El::LOWER && row < col))
                         continue;

                     T val = values[j];
                     c[i * ldc + row] += alpha * val * b[i * ldb + col];
                     if (row != col)
                         c[i * ldc + col] += alpha * val * b[i * ldb + row];
                 }
    }

    // TODO: RIGHT
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const sparse_matrix_t<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& C) {

    base::Symm(side, uplo, alpha, A, B, T(0.0), C);
}

} } // namespace skylark::base

#endif // SKYLARK_SYMM_HPP
