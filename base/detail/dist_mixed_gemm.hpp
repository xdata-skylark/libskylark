#ifndef SKYLARK_DIST_MIXED_GEMM_HPP
#define SKYLARK_DIST_MIXED_GEMM_HPP

#include <boost/mpi.hpp>

#include "../exception.hpp"
#include "../sparse_star_vr_matrix.hpp"
#include "../sparse_vc_star_matrix.hpp"


namespace skylark { namespace base {

// sparse(VC/STAR) * dense(VC/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::VC, El::STAR> &B,
          value_type beta,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    //TODO:
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
}

// sparse(VC/STAR) * dense(VC/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::VC, El::STAR> &B,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    base::Gemm(oA, oB, alpha, A, B, value_type(0.0), C);
}


// sparse(VC/STAR) * dense(VC/STAR) -> dense(STAR/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::VC, El::STAR> &B,
          value_type beta,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    if ((oA == El::ADJOINT || oA == El::TRANSPOSE) && oB == El::NORMAL) {
        boost::mpi::communicator comm(C.Grid().Comm().comm,
            boost::mpi::comm_attach);

        int size = comm.size();
        base::Gemm(oA, oB, alpha, A.locked_matrix(), B.LockedMatrix(),
            beta / value_type(size), C.Matrix());
        boost::mpi::all_reduce (comm,
            boost::mpi::inplace_t<value_type *>(C.Buffer()),
            C.LDim() * C.LocalWidth(), std::plus<value_type>());
    } else
        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
}

// sparse(VC/STAR) * dense(VC/STAR) -> dense(STAR/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::VC, El::STAR> &B,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    base::Gemm(oA, oB, alpha, A, B, value_type(0.0), C);
}


// dense(VC/STAR) * sparse(VC/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const El::DistMatrix<value_type, El::VC, El::STAR> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          value_type beta,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    //TODO
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
}

// dense(VC/STAR) * sparse(VC/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const El::DistMatrix<value_type, El::VC, El::STAR> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    base::Gemm(oA, oB, alpha, A, B, value_type(0.0), C);
}


// dense(STAR/STAR) * sparse(VC/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const El::DistMatrix<value_type, El::STAR, El::STAR> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          value_type beta,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    //TODO:
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
}

// dense(STAR/STAR) * sparse(VC/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const El::DistMatrix<value_type, El::STAR, El::STAR> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    base::Gemm(oA, oB, alpha, A, B, value_type(0.0), C);
}


// dense(VC/STAR) * sparse(VC/STAR) -> dense(STAR/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const El::DistMatrix<value_type, El::VC, El::STAR> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          value_type beta,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    //TODO:
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
}

// dense(VC/STAR) * sparse(VC/STAR) -> dense(STAR/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const El::DistMatrix<value_type, El::VC, El::STAR> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    base::Gemm(oA, oB, alpha, A, B, value_type(0.0), C);
}



// sparse(VC/STAR) * dense(STAR/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::STAR, El::STAR> &B,
          value_type beta,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    assert(A.is_finalized());

    const int* indptr  = A.indptr();
    const int* indices = A.indices();
    const value_type *values = A.locked_values();

    int k = A.local_width();
    int n = B.Width();
    int m = B.Height();

    if (oA == El::ADJOINT && std::is_same<value_type, El::Base<value_type> >::value)
        oA = El::TRANSPOSE;

    if (oB == El::ADJOINT && std::is_same<value_type, El::Base<value_type> >::value)
        oB = El::TRANSPOSE;

    // NN
    if (oA == El::NORMAL && oB == El::NORMAL) {
        base::Gemm(oA, oB, alpha, A.locked_matrix(), B.LockedMatrix(),
            beta, C.Matrix());
        /*
        El::Scale(beta, C);

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int i = 0; i < n; i++)
            for(int col = 0; col < k; col++) {
                int g_col = A.global_col(col);
                value_type tmp_val = alpha * B.Get(g_col, i);
                for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                    assert(indices[j] < C.LocalHeight());
                    assert(i < C.LocalWidth());
                    C.UpdateLocal(indices[j], i, values[j] * tmp_val);
                    //FIXME why is ^^ getting into a double free corruption?
                    //      The asserts hold, and the memory should be reserved?
                }
            }
        */
    }

    // NT
    if (oA == El::NORMAL && (oB == El::TRANSPOSE /*|| oB == El::ADJOINT*/)) {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }

    // TN
    if (oA == El::TRANSPOSE && oB == El::NORMAL) {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }

    // TT
    if (oA == El::TRANSPOSE && (oB == El::TRANSPOSE /*|| oB == El::ADJOINT*/)) {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }
}

// sparse(VC/STAR) * dense(STAR/STAR) -> dense(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::STAR, El::STAR> &B,
          El::DistMatrix<value_type, El::VC, El::STAR> &C) {

    base::Gemm(oA, oB, alpha, A, B, value_type(0.0), C);
}



// sparse(STAR/VR) * dense(VC/STAR) -> dense(STAR/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_star_vr_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::VC, El::STAR> &B,
          value_type beta,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    assert(A.is_finalized());

    const int* indptr  = A.indptr();
    const int* indices = A.indices();
    const value_type *values = A.locked_values();

    int k = A.local_width();
    int n = B.Width();
    int m = B.Height();

    if (oA == El::ADJOINT && std::is_same<value_type, El::Base<value_type> >::value)
        oA = El::TRANSPOSE;

    if (oB == El::ADJOINT && std::is_same<value_type, El::Base<value_type> >::value)
        oB = El::TRANSPOSE;

    // NN
    if (oA == El::NORMAL && oB == El::NORMAL) {

        El::Scale(beta, C);

#if SKYLARK_HAVE_OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n; i++)
            for(int col = 0; col < k; col++) {
                int g_col = A.global_col(col);
                for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                    int row = A.global_row(indices[j]);
                    value_type val = values[j];
                    C.Update(row, i, alpha * val * B.Get(g_col, i));
                }
            }
    } else {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());
    }

    //FIXME: collect C
}


// sparse(STAR/VR) * dense(STAR/STAR) -> sparse(VC/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_vc_star_matrix_t<value_type> &A,
          const El::DistMatrix<value_type, El::STAR, El::STAR> &B,
          value_type beta,
          sparse_vc_star_matrix_t<value_type> &C) {

    assert(A.is_finalized());

    //FIXME that is currently a problem: C can only be non-finalized!
    assert(!C.is_finalized());

    //FIXME: use sparse GEMMs from above or create a new one

    const int* indptr  = A.indptr();
    const int* indices = A.indices();
    const value_type *values = A.locked_values();

    //FIXME: what is k ??
    int k = A.local_width();
    //int k = A.width();
    int n = B.Width();
    int m = B.Height();

    if (oA == El::ADJOINT && std::is_same<value_type, El::Base<value_type> >::value)
        oA = El::TRANSPOSE;

    if (oB == El::ADJOINT && std::is_same<value_type, El::Base<value_type> >::value)
        oB = El::TRANSPOSE;

    // NN
    if (oA == El::NORMAL && oB == El::NORMAL) {

        //FIXME
        //El::Scale(beta, C);

#if SKYLARK_HAVE_OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n; i++)
            for(int col = 0; col < k; col++) {
                int g_col = A.global_col(col);
                for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                    int row = A.global_row(indices[j]);
                    value_type val = values[j];
                    //FIXME: does not work in parallel, we can only update local (row, i)
                    C.queue_update(row, i, alpha * val * B.Get(g_col, i));
                }
            }
    }

    // NT
    if (oA == El::NORMAL && (oB == El::TRANSPOSE /*|| oB == El::ADJOINT*/)) {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

#if 0
        //FIXME:
        //El::Scale(beta, C);

#if SKYLARK_HAVE_OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n; i++)
            for(int col = 0; col < k; col++) {
                int g_col = A.global_col(col);
                for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                    int row = A.global_row(indices[j]);
                    value_type val = values[j];
                    C.queue_update(row, i, alpha * val * B.Get(i, g_col));
                }
            }
#endif
    }

    // TN
    if (oA == El::TRANSPOSE && oB == El::NORMAL) {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

#if 0
#if SKYLARK_HAVE_OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (int j = 0; j < n; j++)
            for(int row = 0; row < k; row++) {
                //FIXME:
                //C.Scale(row, j, beta);
                int g_row = A.global_row(row);
                for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                    int col = A.global_col(indices[l]);
                    value_type val = values[l];
                    C.queue_update(g_row, j,  alpha * val * B.Get(j, col));
                }
            }
#endif
    }

    // TT
    if (oA == El::TRANSPOSE && (oB == El::TRANSPOSE /*|| oB == El::ADJOINT*/)) {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

#if 0
#if SKYLARK_HAVE_OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (int j = 0; j < n; j++)
            for(int row = 0; row < k; row++) {
                //FIXME:
                //C.Scale(row, j, beta);
                int g_row = A.global_row(row);
                for (int l = indptr[row]; l < indptr[row + 1]; l++) {
                    int col = A.global_col(indices[l]);
                    value_type val = values[l];
                    C.queue_update(g_row, j,  alpha * val * B.Get(col, j));
                }
            }
#endif
    }

    C.finalize();
}

// sparse(STAR/Vr) * sparse(VC/STAR) -> dense(STAR/STAR)
template<typename value_type>
void Gemm(El::Orientation oA, El::Orientation oB, value_type alpha,
          const sparse_star_vr_matrix_t<value_type> &A,
          const sparse_vc_star_matrix_t<value_type> &B,
          value_type beta,
          El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    assert(A.is_finalized());
    assert(B.is_finalized());

    //TODO: implement (use local GEMM)
    SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

    // Sum-scatter after local GEMM
    //El::AxpyContract(static_cast<T>(1), C_STAR_STAR, C);
}

} // namespace base
} // namespace skylark

#endif // SKYLARK_DIST_MIXED_GEMM_HPP
