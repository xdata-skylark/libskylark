#ifndef SKYLARK_SYMM_HPP
#define SKYLARK_SYMM_HPP

#include <boost/mpi.hpp>
#include "exception.hpp"
#include "sparse_matrix.hpp"
#include "computed_matrix.hpp"


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

    // RIGHT - TODO: Not tested!
    if (side == El::RIGHT) {
        int k = A.width();
        int n = B.Width();
        int m = A.height();

        El::Scale(beta, C);

        El::Matrix<T> Bc;
        El::Matrix<T> Cc;

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(Cc, Bc)
#       endif
        for(int col = 0; col < n; col++) {
            El::View(Cc, C, 0, col, m, 1);
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];

                if ((uplo == El::UPPER && row > col) ||
                    (uplo == El::LOWER && row < col))
                    continue;

                T val = values[j];
                El::LockedView(Bc, B, 0, row, m, 1);
                El::Axpy(alpha * val, Bc, Cc);
            }
        }
    }
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const sparse_matrix_t<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& C) {

    base::Symm(side, uplo, alpha, A, B, T(0.0), C);
}


/**
 *
 *
 * XXX: In the symmetric case [VC/STAR] = [STAR/VC]
 *
 * FIXME: no uplo support yet (change sparse_dist_matrix_t) and only iterate
 *        over upper/lower part.
 */
template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const skylark::base::sparse_vc_star_matrix_t<T>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& B, T beta,
    El::DistMatrix<T, El::VC, El::STAR>& C) {

    assert(A.is_finalized());

    El::Scale(beta, C);

    //FIXME: there is a visibility issue here??! Check header includes
    //boost::mpi::communicator comm = get_communicator(B);
    boost::mpi::communicator comm =
        boost::mpi::communicator(B.DistComm().comm, boost::mpi::comm_attach);

    const int* indptr  = A.indptr();
    const int* indices = A.indices();
    const T *values = A.locked_values();

    // temporary matrix (only used in RIGHT case)
    El::DistMatrix<T, El::STAR, El::STAR>
        C_STAR_STAR(C.Grid());
    if(side == El::RIGHT) {
        C_STAR_STAR.Resize(C.Height(), C.Width());
        El::Zero(C_STAR_STAR);
    }

    for(int rank = 0; rank < comm.size(); rank++) {

        // broadcast the local values owned by rank, assuming that B is the
        // smallest matrix (assuming A is very tall an skinny)
        El::Matrix<T> tmp;

        // FIXME: is there a more efficient way of doing that using
        //        El::Broadcast?
        size_t width = 0;
        size_t height = 0;
        if(comm.rank() == rank) {
            width  = B.LocalWidth();
            height = B.LocalHeight();
        }
        boost::mpi::broadcast(comm, width, rank);
        boost::mpi::broadcast(comm, height, rank);
        tmp.Resize(height, width);

        if(comm.rank() == rank) tmp = B.LockedMatrix();
        boost::mpi::broadcast(comm, tmp.Buffer(), width * height, rank);

        // LEFT
        if (side == El::LEFT) {

            const int k = A.local_width();
            const int n = tmp.Width();

#if SKYLARK_HAVE_OPENMP
            #pragma omp parallel for
#endif
            for(int i = 0; i < n; i++)
                for(int col = 0; col < k; col++) {
                    int g_col = A.global_col(col);

                    if(B.RowOwner(g_col) != rank) continue;
                    int l_col = g_col / comm.size();
                    // FIXME: why do we need ^^ and B.LocalRow(g_col) does not
                    //        work? Noticed that it produces l_col = 10 for the
                    //        a local matrix [0,9] row inidices!

                    T tmp_val = tmp.Get(l_col, i);
                    for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                        C.UpdateLocal(indices[j], i, alpha * values[j] * tmp_val);
                    }
                }

        } else {

            const int k = A.local_width();
            const int n = tmp.Height();

#if SKYLARK_HAVE_OPENMP
            #pragma omp parallel for
#endif
            for(int i = 0; i < n; i++) {
                int global_row = comm.size() * i + rank;

                for(int col = 0; col < k; col++) {
                    T sum = 0.;
                    for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                        int g_row = A.global_row(indices[j]);
                        sum += alpha * values[j] * tmp.Get(i, g_row);
                    }

                    C_STAR_STAR.UpdateLocal(global_row, col, sum);
                }
            }

            // Reduce-scatter within process grid
            //El::AxpyContract(static_cast<T>(1), C_STAR_STAR, C);
        }
    }

    // Reduce-scatter within process grid
    if (side == El::RIGHT) {
        //El::Print(C_STAR_STAR);
        El::AxpyContract(static_cast<T>(1), C_STAR_STAR, C);
        //El::Print(C);
    }
}

template<typename T>
inline void Symm(El::LeftOrRight side, El::UpperOrLower uplo,
    T alpha, const sparse_vc_star_matrix_t<T>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& B,
    El::DistMatrix<T, El::VC, El::STAR>& C) {

    base::Symm(side, uplo, alpha, A, B, static_cast<T>(1.0), C);
}

} } // namespace skylark::base

#endif // SKYLARK_SYMM_HPP
