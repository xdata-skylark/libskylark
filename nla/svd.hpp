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
 * the output of (A^T * A)^iternum * V (or (A* A^T)^iternum * V).
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

    typedef typename skylark::utility::typer_t<MatrixType>::value_type
        value_type;

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
        base::Gemm(orientation, El::NORMAL, value_type(1.0), A, V, U);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::NORMAL, value_type(1.0), A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::NORMAL, value_type(1.0), A, U, V);
            if (ortho) El::qr::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::NORMAL, value_type(1.0), A, V, U);
    }

    if (vorientation != El::NORMAL && uorientation == El::NORMAL) {
        if (ortho) El::lq::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::ADJOINT, value_type(1.0), A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Gemm(El::ADJOINT, orientation, value_type(1.0), U, A, V);
            if (ortho) El::lq::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::ADJOINT, value_type(1.0), A, V, U);
    }

    if (vorientation == El::NORMAL && uorientation != El::NORMAL) {
        if (ortho) El::qr::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(El::ADJOINT, adjorientation, value_type(1.0), V, A, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::ADJOINT, value_type(1.0), A, U, V);
            if (ortho) El::qr::ExplicitUnitary(V);
        }
        base::Gemm(El::ADJOINT, adjorientation, value_type(1.0), V, A, U);
    }

    if (vorientation != El::NORMAL && uorientation != El::NORMAL) {
        if (ortho) El::lq::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(El::NORMAL, adjorientation, value_type(1.0), V, A, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Gemm(El::NORMAL, orientation, value_type(1.0), U, A, V);
            if (ortho) El::lq::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::ADJOINT, value_type(1.0), A, V, U);
    }
}

/**
 * Power iteration on symmetric matrix from a specific starting vectors
 * (the V input).
 *
 * Performs power iteration on input A .
 * V is both the initial matrix on which the iteration is done, and
 * the outuput of A^iternum * V.
 *
 * \param uplo Whether A is stored in upper or lower part. Might be not-relevant
 *             for certain MatrixType.
 * \param orientation Whether to hold V in transpose or not.
 * \param A input matrix
 * \param V input starting vector, and output of iteration
 * \param iternum how many iterations to do
 * \param otho whether to orthonormalize after every multipication.
 */
template<typename MatrixType, typename VType>
void SymmetricPowerIteration(El::UpperOrLower uplo, El::Orientation vorientation,
    const MatrixType &A, VType &V, int iternum, bool ortho = false) {

    typedef typename skylark::utility::typer_t<MatrixType>::value_type
        value_type;

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef VType v_type;

    index_t n = base::Width(A);
    index_t k = (vorientation == El::NORMAL) ? base::Width(V) : base::Height(V);

    VType U;
    if (vorientation == El::NORMAL)
        U.Resize(n, k);
    else
        U.Resize(k, n);

    if (vorientation == El::NORMAL) {

        for(int i = 0; i < (iternum / 2); i++) {
            if (ortho) El::qr::ExplicitUnitary(V);
            base::Symm(El::LEFT, uplo, value_type(1.0), A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Symm(El::LEFT, uplo, value_type(1.0), A, U, V);
        }
        if (iternum % 2 == 1) {
            if (ortho) El::qr::ExplicitUnitary(V);
            base::Symm(El::LEFT, uplo, value_type(1.0), A, V, U);
            El::Copy(U, V);
        }

    }

    if (vorientation != El::NORMAL) {

        base::Symm(El::LEFT, uplo, value_type(1.0), A, V, V);
        for(int i = 0; i < (iternum / 2); i++) {
            if (ortho) El::lq::ExplicitUnitary(V);
            base::Symm(El::RIGHT, uplo, value_type(1.0), A, V, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Symm(El::RIGHT, uplo, value_type(1.0), A, U, V);
        }
        if (iternum % 2 == 1) {
            if (ortho) El::lq::ExplicitUnitary(V);
            base::Symm(El::LEFT, uplo, value_type(1.0), A, V, U);
            El::Copy(U, V);
        }
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

    typedef typename skylark::utility::typer_t<InputType>::value_type
        value_type;

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
                El::Trsm(El::RIGHT, El::UPPER, El::NORMAL, El::NON_UNIT,
                    value_type(1.0), R, V);
            } else {
                // The above computation, while mathemetically correct for
                // any number of iterations, is not robust enough numerically
                // when number of power iteration is greater than 0.
                El::qr::ExplicitUnitary(Q);
                base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), A, Q, V);
            }
        }

        /** Compute factorization & truncate to rank */
        VType B;
        El::SVD(V, S, B);
        S.Resize(rank, 1); V.Resize(n, rank);
        VType B1 = base::ColumnView(B, 0, rank);
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), Q, B1, U);
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
            base::Gemm(El::NORMAL, El::ADJOINT, value_type(1.0), A, Q, U);
        }

        /** Compute factorization & truncate to rank */
        UType B;
        El::SVD(U, S, B);
        S.Resize(rank, 1); U.Resize(m, rank);
        VType B1 = base::ColumnView(B, 0, rank);
        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), Q, B1, V);
    }
}

/**
 * Approximate SVD computation.
 *
 * Compute an approximate SVD-like decomposition of the input A (symmetric, n-by-n).
 * That is compute V (n-by-rank), S (rank-by-rank) such
 * that A ~= V * S * V^T. S is diagonal, with positive values,  V has
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
template <typename InputType, typename VType, typename SType>
void ApproximateSymmetricSVD(El::UpperOrLower uplo,
    InputType &A, VType &V, SType &S, int rank,
    base::context_t& context,
    approximate_svd_params_t params = approximate_svd_params_t()) {

    typedef typename skylark::utility::typer_t<InputType>::value_type
        value_type;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    int n = base::Width(A);

    /**
     * Sanity check: is the matrix even square?
     */
    if (base::Height(A) != n) {
        std::string msg = "Matrix is not square (so is not symmetric)";
        if (log_lev1)
            params.log_stream << msg << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(msg));
    }

    /**
     * Check if sizes match.
     */
    if (rank > n) {
        std::string msg = "Incompatible matrix dimensions and target rank";
        if (log_lev1)
            params.log_stream << msg << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(msg));
    }

    int k = std::max(rank, std::min(n,
            params.oversampling_ratio * rank +
            params.oversampling_additive));

    /** Apply sketch transformation on the input matrix */
    VType U(n, k), B, W;
    V.Resize(n, k);
    sketch::JLT_t<InputType, VType> Omega(n, k, context);
    Omega.apply(A, V, sketch::rowwise_tag());

    /** Power iteration */
    SymmetricPowerIteration(uplo, El::NORMAL, A, V,params.num_iterations,
        !params.skip_qr);

    /** Schur-Rayleigh-Ritz (with SVD), aka factorize & truncate to rank */
    El::qr::ExplicitUnitary(V);
    base::Symm(El::LEFT, uplo, value_type(1.0), A, V, U);
    base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), U, V, B);
    El::HermitianEig(uplo, B, S, W, El::DESCENDING);
    S.Resize(rank, 1);
    VType W1 = base::ColumnView(W, 0, rank);
    base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), V, W1, U);
    El::Copy(U, V);
}

} } /** namespace skylark::nla */

#endif /** SKYLARK_APPROXIMATE_SVD_HPP */
