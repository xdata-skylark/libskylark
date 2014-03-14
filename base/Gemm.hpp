#ifndef SKYLARK_GEMM_HPP
#define SKYLARK_GEMM_HPP

// Defines a generic Gemm function that recieves both dense and sparse matrices.


#if SKYLARK_HAVE_ELEMENTAL

namespace skylark { namespace base {

/**
 * Rename the elemental Gemm function, so that we have unified access.
 */

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::Matrix<T>& A, const elem::Matrix<T>& B,
    T beta, elem::Matrix<T>& C) {
    elem::Gemm(oA, oB, alpha, A, B, beta, C);
}

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::Matrix<T>& A, const elem::Matrix<T>& B,
    elem::Matrix<T>& C) {
    elem::Gemm(oA, oB, alpha, A, B, C);
}

template<typename T,
         elem::Dist AC, elem::Dist AR,
         elem::Dist BC, elem::Dist BR,
         elem::Dist CC, elem::Dist CR>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::DistMatrix<T, AC, AR>& A,
    const elem::DistMatrix<T, BC, BR>& B,
    T beta, elem::DistMatrix<T, CC, CR>& C) {
    elem::Gemm(oA, oB, alpha, A, B, beta, C);
}

template<typename T,
         elem::Dist AC, elem::Dist AR,
         elem::Dist BC, elem::Dist BR,
         elem::Dist CC, elem::Dist CR>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::DistMatrix<T, AC, AR>& A,
    const elem::DistMatrix<T, BC, BR>& B,
    elem::DistMatrix<T, CC, CR>& C) {
    elem::Gemm(oA, oB, alpha, A, B, C);
}

/**
 * Gemm between mixed elemental, sparse input. Output is dense elemental.
 */

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::Matrix<T>& A, const sparse_matrix_t<T>& B,
    T beta, elem::Matrix<T>& C) {
    // TODO verify sizes etc.

    const std::vector<int> &indptr = B.locked_indptr();
    const std::vector<int> &indices = B.locked_indices();
    const std::vector<T> &values = B.locked_values();

    elem::Scal(beta, C);

    int m = A.Height();
    int n = B.Width();

    elem::Matrix<T> Ac;
    elem::Matrix<T> Cc;

    // NN
    if (oA == elem::NORMAL && oB == elem::NORMAL) {
#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(Cc, Ac)
#       endif
        for(int col = 0; col < n; col++) {
            elem::View(Cc, C, 0, col, m, 1);
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];
                T val = values[j];
                elem::LockedView(Ac, A, 0, row, m, 1);
                elem::Axpy(alpha * val, Ac, Cc);
            }
        }
    }

    // NT
    if (oA == elem::NORMAL && oB == elem::TRANSPOSE) {
        // Now, we simply think of B has being in CSR mode...
        int row = 0;
        for(int row = 0; row < n; row++) {
            elem::LockedView(Ac, A, 0, row, m, 1);
#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for private(Cc, Ac)
#           endif
            for (int j = indptr[row]; j < indptr[row + 1]; j++) {
                int col = indices[j];
                T val = values[j];
                elem::View(Cc, C, 0, col, m, 1);
                elem::Axpy(alpha * val, Ac, Cc);
            }
        }
    }
}

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const sparse_matrix_t<T>& A, const elem::Matrix<T>& B,
    T beta, elem::Matrix<T>& C) {
    // TODO verify sizes etc.

    const std::vector<int> &indptr = A.locked_indptr();
    const std::vector<int> &indices = A.locked_indices();
    const std::vector<T> &values = A.locked_values();

    elem::Scal(beta, C);

    int n = B.Width();

    if (oA == elem::NORMAL && oB == elem::NORMAL) {
        // TODO!
    }
}

} }

#endif // SKYLARK_HAVE_ELEMENTAL

#endif // SKYLARK_GEMM_HPP
