#ifndef SKYLARK_GEMM_HPP
#define SKYLARK_GEMM_HPP

#include <boost/mpi.hpp>

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

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::DistMatrix<T>& A, const elem::DistMatrix<T>& B,
    T beta, elem::DistMatrix<T>& C) {
    elem::Gemm(oA, oB, alpha, A, B, beta, C);
}

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::DistMatrix<T>& A, const elem::DistMatrix<T>& B,
    elem::DistMatrix<T>& C) {
    elem::Gemm(oA, oB, alpha, A, B, C);
}

/**
 * The following combination is not offered by Elemental, but is useful for us.
 * We implement it partially.
 */
template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::DistMatrix<T, elem::VC, elem::STAR>& A, 
    const elem::DistMatrix<T, elem::VC, elem::STAR>& B,
    T beta, elem::DistMatrix<T, elem::STAR, elem::STAR>& C) {
    // TODO verify sizes etc.

    if ((oA == elem::TRANSPOSE || oA == elem::ADJOINT) && oB == elem::NORMAL) {
        elem::Matrix<T> Clocal(C.Height(), C.Width(), C.Matrix().LDim());
        elem::Gemm(oA, elem::NORMAL,
            alpha, A.LockedMatrix(), B.LockedMatrix(),
            beta, Clocal);
        boost::mpi::communicator comm(C.DistComm(), boost::mpi::comm_attach);
        boost::mpi::all_reduce(comm,
            Clocal.Buffer(), Clocal.MemorySize(), C.Buffer(),
            std::plus<T>());
    } else {
        // Not supported!  TODO exception checking!
    }
}

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::DistMatrix<T, elem::VC, elem::STAR>& A, 
    const elem::DistMatrix<T, elem::VC, elem::STAR>& B,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& C) {

    int C_height = (oA == elem::NORMAL ? A.Height() : A.Width());
    int C_width = (oB == elem::NORMAL ? B.Width() : B.Height());
    elem::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}

/**
 * Gemm between mixed elemental, sparse input. Output is dense elemental.
 */

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const elem::Matrix<T>& A, const sparse_matrix_t<T>& B,
    T beta, elem::Matrix<T>& C) {
    // TODO verify sizes etc.

    const int* indptr = B.indptr();
    const int* indices = B.indices();
    const T *values = B.locked_values();

    int k = A.Width();
    int n = B.Width();
    int m = A.Height();

    // NN
    if (oA == elem::NORMAL && oB == elem::NORMAL) {

        elem::Scal(beta, C);

        elem::Matrix<T> Ac;
        elem::Matrix<T> Cc;

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

        elem::Scal(beta, C);

        elem::Matrix<T> Ac;
        elem::Matrix<T> Cc;

        // Now, we simply think of B has being in CSR mode...
        int row = 0;
        for(int row = 0; row < n; row++) {
            elem::LockedView(Ac, A, 0, row, m, 1);
#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for private(Cc)
#           endif
            for (int j = indptr[row]; j < indptr[row + 1]; j++) {
                int col = indices[j];
                T val = values[j];
                elem::View(Cc, C, 0, col, m, 1);
                elem::Axpy(alpha * val, Ac, Cc);
            }
        }
    }


    // TN - TODO: Not tested!
    if (oA == elem::TRANSPOSE && oB == elem::NORMAL) {
        double *c = C.Buffer();
        int ldc = C.LDim();

        const double *a = A.LockedBuffer();
        int lda = A.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for (int j = 0; j < n; j++)
            for(int row = 0; row < k; row++) {
                c[j * ldc + row] *= beta;
                 for (int l = indptr[j]; l < indptr[j + 1]; l++) {
                     int rr = indices[l];
                     T val = values[l];
                     c[j * ldc + row] += val * a[j * lda + rr];
                 }
            }
    }

    // TT - TODO: Not tested!
    if (oA == elem::TRANSPOSE && oB == elem::TRANSPOSE) {
        elem::Scal(beta, C);

        double *c = C.Buffer();
        int ldc = C.LDim();

        const double *a = A.LockedBuffer();
        int lda = A.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int row = 0; row < k; row++)
            for(int rb = 0; rb < n; rb++)
                for (int l = indptr[rb]; l < indptr[rb + 1]; l++) {
                    int col = indices[l];
                    c[col * ldc + row] += values[l] * a[row * lda + rb];
                }
    }
}

template<typename T>
inline void Gemm(elem::Orientation oA, elem::Orientation oB,
    T alpha, const sparse_matrix_t<T>& A, const elem::Matrix<T>& B,
    T beta, elem::Matrix<T>& C) {
    // TODO verify sizes etc.

    const int* indptr = A.indptr();
    const int* indices = A.indices();
    const double *values = A.locked_values();

    int k = A.Width();
    int n = B.Width();
    int m = B.Height();

    // NN
    if (oA == elem::NORMAL && oB == elem::NORMAL) {

        elem::Scal(beta, C);

        double *c = C.Buffer();
        int ldc = C.LDim();

        const double *b = B.LockedBuffer();
        int ldb = B.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int i = 0; i < n; i++)
            for(int col = 0; col < k; col++)
                 for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                     int row = indices[j];
                     T val = values[j];
                     c[i * ldc + row] += alpha * val * b[i * ldb + col];
                 }
    }

    // NT
    if (oA == elem::NORMAL && oB == elem::TRANSPOSE) {

        elem::Scal(beta, C);

        elem::Matrix<T> Bc;
        elem::Matrix<T> BTr;
        elem::Matrix<T> Cr;

        for(int col = 0; col < k; col++) {
            elem::LockedView(Bc, B, 0, col, m, 1);
            elem::Transpose(Bc, BTr);
#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for private(Cr)
#           endif
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];
                T val = values[j];
                elem::View(Cr, C, row, 0, 1, m);
                elem::Axpy(alpha * val, BTr, Cr);
            }
        }
    }

    // TN - TODO: Not tested!
    if (oA == elem::TRANSPOSE && oB == elem::NORMAL) {
        double *c = C.Buffer();
        int ldc = C.LDim();

        const double *b = B.LockedBuffer();
        int ldb = B.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for (int j = 0; j < n; j++)
            for(int row = 0; row < k; row++) {
                c[j * ldc + row] *= beta;
                 for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                     int col = indices[l];
                     T val = values[l];
                     c[j * ldc + row] += val * b[j * ldb + col];
                 }
            }
    }

    // TT - TODO: Not tested!
    if (oA == elem::TRANSPOSE && oB == elem::TRANSPOSE) {

        elem::Scal(beta, C);

        elem::Matrix<T> Bc;
        elem::Matrix<T> BTr;
        elem::Matrix<T> Cr;

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(Cr, Bc, BTr)
#       endif
        for(int row = 0; row < k; row++) {
            elem::View(Cr, C, row, 0, 1, m);
            for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                int col = indices[l];
                T val = values[l];
                elem::LockedView(Bc, B, 0, col, m, 1);
                elem::Transpose(Bc, BTr);
                elem::Axpy(alpha * val, BTr, Cr);
            }
        }
    }
}

} } // namespace skylark::base

#endif // SKYLARK_HAVE_ELEMENTAL

#endif // SKYLARK_GEMM_HPP
