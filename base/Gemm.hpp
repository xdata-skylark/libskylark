#ifndef SKYLARK_GEMM_HPP
#define SKYLARK_GEMM_HPP

#include <boost/mpi.hpp>
#include "exception.hpp"
#include "sparse_matrix.hpp"
#include "computed_matrix.hpp"
#include "../utility/typer.hpp"

#include "Gemm_detail.hpp"

// Defines a generic Gemm function that receives both dense and sparse matrices.

namespace skylark { namespace base {

/**
 * Rename the Elemental Gemm function, so that we have unified access.
 */

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::Matrix<T>& A, const El::Matrix<T>& B,
    T beta, El::Matrix<T>& C) {
    El::Gemm(oA, oB, alpha, A, B, beta, C);
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::Matrix<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& C) {
    El::Gemm(oA, oB, alpha, A, B, C);
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::STAR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    T beta, El::DistMatrix<T, El::STAR, El::STAR>& C) {
    El::Gemm(oA, oB, alpha, A.LockedMatrix(), B.LockedMatrix(), beta, C.Matrix());
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::STAR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& C) {
    El::Gemm(oA, oB, alpha, A.LockedMatrix(), B.LockedMatrix(), C.Matrix());
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T>& A, const El::DistMatrix<T>& B,
    T beta, El::DistMatrix<T>& C) {
    El::Gemm(oA, oB, alpha, A, B, beta, C);
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T>& A, const El::DistMatrix<T>& B,
    El::DistMatrix<T>& C) {
    El::Gemm(oA, oB, alpha, A, B, C);
}

/**
 * The following combinations is not offered by Elemental, but are useful for us.
 * We implement them partially.
 */

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& B,
    T beta, El::DistMatrix<T, El::STAR, El::STAR>& C) {
    // TODO verify sizes etc.

    if ((oA == El::TRANSPOSE || oA == El::ADJOINT) && oB == El::NORMAL) {
        boost::mpi::communicator comm(C.Grid().Comm().comm, 
            boost::mpi::comm_attach);
        El::Matrix<T> Clocal(C.Matrix());
        El::Gemm(oA, El::NORMAL,
            alpha, A.LockedMatrix(), B.LockedMatrix(),
            beta / T(comm.size()), Clocal);
        boost::mpi::all_reduce(comm,
            Clocal.Buffer(), Clocal.MemorySize(), C.Matrix().Buffer(),
            std::plus<T>());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& C) {

    int C_height = (oA == El::NORMAL ? A.Height() : A.Width());
    int C_width = (oB == El::NORMAL ? B.Width() : B.Height());
    El::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    T beta, El::DistMatrix<T, El::VC, El::STAR>& C) {
    // TODO verify sizes etc.

    if (oA == El::NORMAL && oB == El::NORMAL) {
        El::Gemm(El::NORMAL, El::NORMAL,
            alpha, A.LockedMatrix(), B.LockedMatrix(),
            beta, C.Matrix());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    El::DistMatrix<T, El::VC, El::STAR>& C) {

    int C_height = (oA == El::NORMAL ? A.Height() : A.Width());
    int C_width = (oB == El::NORMAL ? B.Width() : B.Height());
    El::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}


template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VR, El::STAR>& A,
    const El::DistMatrix<T, El::VR, El::STAR>& B,
    T beta, El::DistMatrix<T, El::STAR, El::STAR>& C) {
    // TODO verify sizes etc.

    if ((oA == El::TRANSPOSE || oA == El::ADJOINT) && oB == El::NORMAL) {
        boost::mpi::communicator comm(C.Grid().Comm(), boost::mpi::comm_attach);
        El::Matrix<T> Clocal(C.Matrix());
        El::Gemm(oA, El::NORMAL,
            alpha, A.LockedMatrix(), B.LockedMatrix(),
            beta / T(comm.size()), Clocal);
        boost::mpi::all_reduce(comm,
            Clocal.Buffer(), Clocal.MemorySize(), C.Matrix().Buffer(),
            std::plus<T>());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VR, El::STAR>& A,
    const El::DistMatrix<T, El::VR, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& C) {

    int C_height = (oA == El::NORMAL ? A.Height() : A.Width());
    int C_width = (oB == El::NORMAL ? B.Width() : B.Height());
    El::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    T beta, El::DistMatrix<T, El::VR, El::STAR>& C) {
    // TODO verify sizes etc.

    if (oA == El::NORMAL && oB == El::NORMAL) {
        El::Gemm(El::NORMAL, El::NORMAL,
            alpha, A.LockedMatrix(), B.LockedMatrix(),
            beta, C.Matrix());
    } else {
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::DistMatrix<T, El::VR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    El::DistMatrix<T, El::VR, El::STAR>& C) {

    int C_height = (oA == El::NORMAL ? A.Height() : A.Width());
    int C_width = (oB == El::NORMAL ? B.Width() : B.Height());
    El::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}

/**
 * Gemm between mixed Elental, sparse input. Output is dense Elemental.
 */

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::Matrix<T>& A, const sparse_matrix_t<T>& B,
    T beta, El::Matrix<T>& C) {
    // TODO verify sizes etc.

    const int* indptr = B.indptr();
    const int* indices = B.indices();
    const T *values = B.locked_values();

    if (oA == El::ADJOINT && std::is_same<T, El::Base<T> >::value)
        oA = El::TRANSPOSE;

    if (oB == El::ADJOINT && std::is_same<T, El::Base<T> >::value)
        oB = El::TRANSPOSE;

    if (oA == El::ADJOINT || oB == El::ADJOINT)
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

    // NN
    if (oA == El::NORMAL && oB == El::NORMAL) {
        int k = A.Width();
        int n = B.width();
        int m = A.Height();

        El::Scale(beta, C);

        El::Matrix<T> Ac;
        El::Matrix<T> Cc;

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(Cc, Ac)
#       endif
        for(int col = 0; col < n; col++) {
            El::View(Cc, C, 0, col, m, 1);
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];
                T val = values[j];
                El::LockedView(Ac, A, 0, row, m, 1);
                El::Axpy(alpha * val, Ac, Cc);
            }
        }
    }

    // NT
    if (oA == El::NORMAL && oB == El::TRANSPOSE) {
        int k = A.Width();
        int n = B.width();
        int m = A.Height();

        El::Scale(beta, C);

        El::Matrix<T> Ac;
        El::Matrix<T> Cc;

        // Now, we simply think of B has being in CSR mode...
        int row = 0;
        for(int row = 0; row < n; row++) {
            El::LockedView(Ac, A, 0, row, m, 1);
#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for private(Cc)
#           endif
            for (int j = indptr[row]; j < indptr[row + 1]; j++) {
                int col = indices[j];
                T val = values[j];
                El::View(Cc, C, 0, col, m, 1);
                El::Axpy(alpha * val, Ac, Cc);
            }
        }
    }


    // TN - TODO: Not tested!
    if (oA == El::TRANSPOSE && oB == El::NORMAL) {
        int k = A.Height();
        int n = B.width();
        int m = A.Width();

        T *c = C.Buffer();
        int ldc = C.LDim();

        const T *a = A.LockedBuffer();
        int lda = A.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for (int j = 0; j < n; j++)
            for(int row = 0; row < m; row++) {
                c[j * ldc + row] *= beta;
                 for (int l = indptr[j]; l < indptr[j + 1]; l++) {
                     int rr = indices[l];
                     T val = values[l];
                     c[j * ldc + row] += val * a[row * lda + rr];
                 }
            }
    }

    // TT - TODO: Not tested!
    if (oA == El::TRANSPOSE && oB == El::TRANSPOSE) {
        int k = A.Height();
        int n = B.width();
        int m = A.Width();

        El::Scale(beta, C);

        T *c = C.Buffer();
        int ldc = C.LDim();

        const T *a = A.LockedBuffer();
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
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const sparse_matrix_t<T>& A, const El::Matrix<T>& B,
    T beta, El::Matrix<T>& C) {
    // TODO verify sizes etc.

    const int* indptr = A.indptr();
    const int* indices = A.indices();
    const T *values = A.locked_values();

    int k = A.width();
    int n = B.Width();
    int m = B.Height();

    if (oA == El::ADJOINT && std::is_same<T, El::Base<T> >::value)
        oA = El::TRANSPOSE;

    if (oB == El::ADJOINT && std::is_same<T, El::Base<T> >::value)
        oB = El::TRANSPOSE;

    // NN
    if (oA == El::NORMAL && oB == El::NORMAL) {

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
                     T val = values[j];
                     c[i * ldc + row] += alpha * val * b[i * ldb + col];
                 }
    }

    // NT
    if (oA == El::NORMAL && (oB == El::TRANSPOSE || oB == El::ADJOINT)) {

        El::Scale(beta, C);

        El::Matrix<T> Bc;
        El::Matrix<T> BTr;
        El::Matrix<T> Cr;

        for(int col = 0; col < k; col++) {
            El::LockedView(Bc, B, 0, col, m, 1);
            El::Transpose(Bc, BTr, oB == El::ADJOINT);
#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for private(Cr)
#           endif
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];
                T val = values[j];
                El::View(Cr, C, row, 0, 1, m);
                El::Axpy(alpha * val, BTr, Cr);
            }
        }
    }

    // TN - TODO: Not tested!
    if (oA == El::TRANSPOSE && oB == El::NORMAL) {
        T *c = C.Buffer();
        int ldc = C.LDim();

        const T *b = B.LockedBuffer();
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

    // AN - TODO: Not tested!
    if (oA == El::ADJOINT && oB == El::NORMAL) {
        T *c = C.Buffer();
        int ldc = C.LDim();

        const T *b = B.LockedBuffer();
        int ldb = B.LDim();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for (int j = 0; j < n; j++)
            for(int row = 0; row < k; row++) {
                c[j * ldc + row] *= beta;
                 for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                     int col = indices[l];
                     T val = El::Conj(values[l]);
                     c[j * ldc + row] += val * b[j * ldb + col];
                 }
            }
    }


    // TT - TODO: Not tested!
    if (oA == El::TRANSPOSE && (oB == El::TRANSPOSE || oB == El::ADJOINT)) {

        El::Scale(beta, C);

        El::Matrix<T> Bc;
        El::Matrix<T> BTr;
        El::Matrix<T> Cr;

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(Cr, Bc, BTr)
#       endif
        for(int row = 0; row < k; row++) {
            El::View(Cr, C, row, 0, 1, m);
            for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                int col = indices[l];
                T val = values[l];
                El::LockedView(Bc, B, 0, col, m, 1);
                El::Transpose(Bc, BTr, oB == El::ADJOINT);
                El::Axpy(alpha * val, BTr, Cr);
            }
        }
    }

    // AT - TODO: Not tested!
    if (oA == El::ADJOINT && (oB == El::TRANSPOSE || oB == El::ADJOINT)) {

        El::Scale(beta, C);

        El::Matrix<T> Bc;
        El::Matrix<T> BTr;
        El::Matrix<T> Cr;

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(Cr, Bc, BTr)
#       endif
        for(int row = 0; row < k; row++) {
            El::View(Cr, C, row, 0, 1, m);
            for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                int col = indices[l];
                T val = El::Conj(values[l]);
                El::LockedView(Bc, B, 0, col, m, 1);
                El::Transpose(Bc, BTr, oB == El::ADJOINT);
                El::Axpy(alpha * val, BTr, Cr);
            }
        }
    }
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const sparse_matrix_t<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& C) {
    int C_height = (oA == El::NORMAL ? A.height() : A.width());
    int C_width = (oB == El::NORMAL ? B.Width() : B.Height());
    El::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}

template<typename T>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    T alpha, const El::Matrix<T>& A, const sparse_matrix_t<T>& B,
    El::Matrix<T>& C) {
    int C_height = (oA == El::NORMAL ? A.Height() : A.Width());
    int C_width = (oB == El::NORMAL ? B.width() : B.height());
    El::Zeros(C, C_height, C_width);
    base::Gemm(oA, oB, alpha, A, B, T(0), C);
}

#if SKYLARK_HAVE_COMBBLAS
/**
 * Mixed GEMM for Elental and CombBLAS matrices. For a distributed Elental
 * input matrix, the output has the same distribution.
 */

/// Gemm for distCombBLAS x distElental(* / *) -> distElental (SOMETHING / *)
template<typename index_type, typename value_type, El::Distribution col_d>
void Gemm(El::Orientation oA, El::Orientation oB, double alpha,
          const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
          const El::DistMatrix<value_type, El::STAR, El::STAR> &B,
          double beta,
          El::DistMatrix<value_type, col_d, El::STAR> &C) {

    if(oA == El::NORMAL && oB == El::NORMAL) {

        if(A.getnol() != B.Height())
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg("Gemm: Dimensions do not agree"));

        if(A.getnrow() != C.Height())
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg("Gemm: Dimensions do not agree"));

        if(B.Width() != C.Width())
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg("Gemm: Dimensions do not agree"));

        //XXX: simple heuristic to decide what to communicate (improve!)
        //     or just if A.getncol() < B.Width..
        if(A.getnnz() < B.Height() * B.Width())
            detail::outer_panel_mixed_gemm_impl_nn(alpha, A, B, beta, C);
        else
            detail::inner_panel_mixed_gemm_impl_nn(alpha, A, B, beta, C);
    }
}

/// Gemm for distCombBLAS x distElental(SOMETHING / *) -> distElental (* / *)
template<typename index_type, typename value_type, El::Distribution col_d>
void Gemm(El::Orientation oA, El::Orientation oB, double alpha,
          const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
          const El::DistMatrix<value_type, col_d, El::STAR> &B,
          double beta,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    if(oA == El::TRANSPOSE && oB == El::NORMAL) {

        if(A.getrow() != B.Height())
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg("Gemm: Dimensions do not agree"));

        if(A.getncol() != C.Height())
            SKYLARK_THROW_EXCEPTION (
                    base::combblas_exception()
                    << base::error_msg("Gemm: Dimensions do not agree"));

        if(B.Width() != C.Width())
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg("Gemm: Dimensions do not agree"));

        detail::outer_panel_mixed_gemm_impl_tn(alpha, A, B, beta, C);
    }

}

#endif // SKYLARK_HAVE_COMBBLAS

/* All combinations with computed matrix */

template<typename CT, typename RT, typename OT>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    typename utility::typer_t<OT>::value_type alpha, const computed_matrix_t<CT>& A,
    const RT& B, typename utility::typer_t<OT>::value_type beta, OT& C) {
    base::Gemm(oA, oB, alpha, A.materialize(), B, beta, C);
}

template<typename CT, typename RT, typename OT>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    typename utility::typer_t<OT>::value_type alpha, const computed_matrix_t<CT>& A,
    const RT& B, OT& C) {
    base::Gemm(oA, oB, alpha, A.materialize(), B, C);
}

template<typename CT, typename RT, typename OT>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    typename utility::typer_t<OT>::value_type alpha, const RT& A,
    const computed_matrix_t<CT>& B,
    typename utility::typer_t<OT>::value_type beta, OT& C) {
    base::Gemm(oA, oB, alpha, A, B.materialize(), beta, C);
}

template<typename CT, typename RT, typename OT>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    typename utility::typer_t<OT>::value_type alpha, const RT& A,
    const computed_matrix_t<CT>& B, OT& C) {
    base::Gemm(oA, oB, alpha, A, B.materialize(), C);
}

template<typename CT1, typename CT2, typename OT>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    typename utility::typer_t<OT>::value_type alpha, const computed_matrix_t<CT1>& A,
    const computed_matrix_t<CT2>& B, typename utility::typer_t<OT>::value_type beta,
    OT& C) {
    base::Gemm(oA, oB, alpha, A.materialize(), B.materialize(), beta, C);
}

template<typename CT1, typename CT2, typename OT>
inline void Gemm(El::Orientation oA, El::Orientation oB,
    typename utility::typer_t<OT>::value_type alpha, const computed_matrix_t<CT1>& A,
    const computed_matrix_t<CT2>& B, OT& C) {
    base::Gemm(oA, oB, alpha, A.materialize(), B.materialize(), C);
}


} } // namespace skylark::base
#endif // SKYLARK_GEMM_HPP
