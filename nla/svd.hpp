#ifndef SKYLARK_APPROXIMATE_SVD_HPP
#define SKYLARK_APPROXIMATE_SVD_HPP

#include <El.hpp>
#include "../sketch/sketch.hpp"

namespace skylark { namespace nla {

/**
 * Power iteration from a specific starting vectors (the V input).
 *
 * Performs power iteration on input A or A^T (depending on orientation).
 * V is both the initial matrix on which the iteration is done, and
 * the outuput of (A^T * A)^iternum * V (or (A* A^T)^iternum * V).
 * U is  an auxilary variable (to specify types), but will also be equal to
 * A * V on output (or A^T * V). ortho specifies whether to orthonomralize
 * after each multipication by A or A^T -- essientally doing a subspace
 * iteration. However, note that U = A * V (or A^T * V) always on output.
 *
 * \param orientation Whether to do on A or A^T.
 * \param vorientation Whether to hold V in tranpose or not.
 * \param uorientation Whether to hold U in tranpose or not.
 * \param A input matrix
 * \param V input starting vector, and output of iteration
 * \param U on output: U = A*V or A^T*V.
 * \param iternum how many iterations to do
 * \param otho whether to orthonormalize after every multipication.
 */
template<typename MatrixType, typename LeftType, typename RightType>
void PowerIteration(El::Orientation orientation, El::Orientation vorientation,
    El::Orientation uorientation, const MatrixType &A,
    RightType &V, LeftType &U,
    int iternum, bool ortho = false) {

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef RightType right_type;
    typedef LeftType left_type;

    index_t m = base::Height(A);
    index_t n = base::Width(A);
    index_t k = (vorientation == El::NORMAL) ? base::Width(V) : base::Height(V);

    El::Orientation adjorientation;
    if (orientation == El::ADJOINT || orientation == El::TRANSPOSE) {
        if (uorientation == El::NORMAL)
            U.Resize(n, k);
        else
            U.Resize(k, n);
        adjorientation = El::NORMAL;
    } else {
        if (uorientation == El::NORMAL)
            U.Resize(m, k);
        else
            U.Resize(k, m);
        adjorientation = El::ADJOINT;
    }

    if (vorientation == El::NORMAL && uorientation == El::NORMAL) {
        if (ortho) El::qr::ExplicitUnitary(V);
        base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::NORMAL, 1.0, A, U, V);
            if (ortho) El::qr::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
    }

    if (vorientation != El::NORMAL && uorientation == El::NORMAL) {
        if (ortho) El::lq::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::ADJOINT, 1.0, A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Gemm(El::ADJOINT, orientation, 1.0, U, A, V);
            if (ortho) El::lq::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::ADJOINT, 1.0, A, V, U);
    }

    if (vorientation == El::NORMAL && uorientation != El::NORMAL) {
        if (ortho) El::qr::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(El::ADJOINT, adjorientation, 1.0, V, A, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::ADJOINT, 1.0, A, U, V);
            if (ortho) El::qr::ExplicitUnitary(V);
        }
        base::Gemm(El::ADJOINT, adjorientation, 1.0, V, A, U);
    }

    if (vorientation != El::NORMAL && uorientation != El::NORMAL) {
        if (ortho) El::lq::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(El::NORMAL, adjorientation, 1.0, V, A, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Gemm(El::NORMAL, orientation, 1.0, U, A, V);
            if (ortho) El::lq::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::ADJOINT, 1.0, A, V, U);
    }
}

/**
 * Parameter structure for approximate SVD
 *
 * oversampling_ratio, oversampling_additive:
 *   given a rank r, the number of columns k in the iterates is
 *   k = oversampling_ratio * r + oversampling_additive
 * num_iterations: number of power iteration to do
 * skip_qr: skip doing QR in every iteration (less accurate).
 */
struct approximate_svd_params_t : public base::params_t {

    int oversampling_ratio, oversampling_additive;
    int num_iterations;
    bool skip_qr;

    approximate_svd_params_t(int oversampling_ratio = 2,
        int oversampling_additive = 0,
        int num_iterations = 0,
        bool skip_qr = false,
        bool am_i_printing = false,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, debug_level),
        oversampling_ratio(oversampling_ratio),
        oversampling_additive(oversampling_additive),
        num_iterations(num_iterations), skip_qr(skip_qr) {};
};

/**
 * Approximate SVD computation.
 *
 * Compute an approximate SVD-like decomposition of the input A (m-by-n).
 * That is compute U (m-by-rank), S (rank-by-rank), and V (n-by-rank) such
 * that A ~= U * S * V^T. S is diagonal, with positive values, U and V have
 * orthonormal columns.
 *
 * Based on:
 *
 * Halko, Martinsson and Tropp
 * Finding Structure with Randomness: Probabilistic Algorithms for Constructing
 * Approximate Matrix Decompositions
 * SIAM Rev., 53(2), 217–288. (72 pages)
 *
 * \param A input matrix
 * \param U output: approximate left-singular vectors
 * \param S output: approximate singular values
 * \param V output: approximate right-singular vectors
 * \param rank target rank
 * \param params parameter strcture
 */
template <typename InputType, typename UType, typename SType, typename VType>
void ApproximateSVD(InputType &A, UType &U, SType &S, VType &V, int rank,
    base::context_t& context,
    approximate_svd_params_t params = approximate_svd_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    int m = base::Height(A);
    int n = base::Width(A);

    /**
     * Check if sizes match.
     */
    if (rank > std::min(m, n)) {
        std::string msg = "Incompatible matrix dimensions and target rank";
        if (log_lev1)
            params.log_stream << msg << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(msg));
    }

    /** Code for m >= n */
    if (m >= n) {
        int k = std::max(rank, std::min(n,
                params.oversampling_ratio * rank +
                params.oversampling_additive));

        /** Apply sketch transformation on the input matrix */
        UType Q(m, k);
        sketch::JLT_t<InputType, UType> Omega(n, k, context);
        Omega.apply(A, Q, sketch::rowwise_tag());

        /** Power iteration */
        PowerIteration(El::ADJOINT, El::NORMAL, El::NORMAL, A, Q, V,
            params.num_iterations, !params.skip_qr);

        if (params.skip_qr) {
            if (params.num_iterations == 0) {
                VType R;
                El::qr::Explicit(Q, R);
                El::Trsm(El::RIGHT, El::UPPER, El::NORMAL, El::NON_UNIT, 1.0,
                    R, V);
            } else {
                // The above computation, while mathemetically correct for
                // any number of iterations, is not robust enough numerically
                // when number of power iteration is greater than 0.
                El::qr::ExplicitUnitary(Q);
                base::Gemm(El::ADJOINT, El::NORMAL, 1.0, A, Q, V);
            }
        }

        /** Compute factorization & truncate to rank */
        VType B;
        El::SVD(V, S, B);
        S.Resize(rank, 1); V.Resize(n, rank);
        VType B1 = base::ColumnView(B, 0, rank);
        base::Gemm(El::NORMAL, El::NORMAL, 1.0, Q, B1, U);
    }

    /** Code for m < n */
    if (m < n) {
        int k = std::max(rank, std::min(m,
                params.oversampling_ratio * rank +
                params.oversampling_additive));

        /** Apply sketch transformation on the input matrix */
        VType Q(k, n);
        sketch::JLT_t<InputType, UType> Omega(m, k, context);
        Omega.apply(A, Q, sketch::columnwise_tag());

        /** Power iteration */
        PowerIteration(El::NORMAL, El::ADJOINT, El::NORMAL, A, Q, U,
            params.num_iterations, !params.skip_qr);

        if (params.skip_qr) {
            // We should be able to do the same trick as for m>=n if
            // num_iteration == 0, but the LQ factorization in Elemental
            // does not work for this... (do not know why).
            El::lq::ExplicitUnitary(Q);
            base::Gemm(El::NORMAL, El::ADJOINT, 1.0, A, Q, U);
        }

        /** Compute factorization & truncate to rank */
        UType B;
        El::SVD(U, S, B);
        S.Resize(rank, 1); U.Resize(m, rank);
        VType B1 = base::ColumnView(B, 0, rank);
        base::Gemm(El::ADJOINT, El::NORMAL, 1.0, Q, B1, V);
    }
}

} } /** namespace skylark::nla */

#endif /** SKYLARK_APPROXIMATE_SVD_HPP */
