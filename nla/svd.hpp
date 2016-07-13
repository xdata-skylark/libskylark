#ifndef SKYLARK_APPROXIMATE_SVD_HPP
#define SKYLARK_APPROXIMATE_SVD_HPP

#include <El.hpp>

#include <algorithm>
#include <string>

#include "../utility/types.hpp"

namespace skylark { namespace nla {

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
        std::string prefix = "",
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, prefix, debug_level),
        oversampling_ratio(oversampling_ratio),
        oversampling_additive(oversampling_additive),
        num_iterations(num_iterations), skip_qr(skip_qr) {}

    approximate_svd_params_t(const boost::property_tree::ptree& json)
        : params_t(json) {
        oversampling_ratio = json.get<int>("oversampling_ratio");
        oversampling_additive = json.get<int>("oversampling_additive");
        num_iterations = json.get<int>("num_iterations");
        skip_qr = json.get<bool>("skip_qr");
    }
};

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
 * \param ortho whether to orthonormalize after every multipication.
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
        base::Gemm(orientation, El::NORMAL, static_cast<value_type>(1.0), A, V, U);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::NORMAL, static_cast<value_type>(1.0), A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::NORMAL, static_cast<value_type>(1.0), A, U, V);
            if (ortho) El::qr::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::NORMAL, static_cast<value_type>(1.0), A, V, U);
    }

    if (vorientation != El::NORMAL && uorientation == El::NORMAL) {
        if (ortho) El::lq::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::ADJOINT, static_cast<value_type>(1.0), A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Gemm(El::ADJOINT, orientation, static_cast<value_type>(1.0), U, A, V);
            if (ortho) El::lq::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::ADJOINT, static_cast<value_type>(1.0), A, V, U);
    }

    if (vorientation == El::NORMAL && uorientation != El::NORMAL) {
        if (ortho) El::qr::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(El::ADJOINT, adjorientation, static_cast<value_type>(1.0), V, A, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::ADJOINT, static_cast<value_type>(1.0), A, U, V);
            if (ortho) El::qr::ExplicitUnitary(V);
        }
        base::Gemm(El::ADJOINT, adjorientation, static_cast<value_type>(1.0), V, A, U);
    }

    if (vorientation != El::NORMAL && uorientation != El::NORMAL) {
        if (ortho) El::lq::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(El::NORMAL, adjorientation, static_cast<value_type>(1.0), V, A, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Gemm(El::NORMAL, orientation, static_cast<value_type>(1.0), U, A, V);
            if (ortho) El::lq::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::ADJOINT, static_cast<value_type>(1.0), A, V, U);
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
 * \param vorientation Whether to hold V in transpose or not.
 * \param A input matrix
 * \param V input starting vector, and output of iteration
 * \param iternum how many iterations to do
 * \param ortho whether to orthonormalize after every multipication.
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
        for (int i = 0; i < (iternum / 2); i++) {
            if (ortho) El::qr::ExplicitUnitary(V);
            base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, V, U);
            if (ortho) El::qr::ExplicitUnitary(U);
            base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, U, V);
        }
        if (iternum % 2 == 1) {
            if (ortho) El::qr::ExplicitUnitary(V);
            base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, V, U);
            El::Copy(U, V);
        }
    }

    if (vorientation != El::NORMAL) {
        base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, V, V);
        for (int i = 0; i < (iternum / 2); i++) {
            if (ortho) El::lq::ExplicitUnitary(V);
            base::Symm(El::RIGHT, uplo, static_cast<value_type>(1.0), A, V, U);
            if (ortho) El::lq::ExplicitUnitary(U);
            base::Symm(El::RIGHT, uplo, static_cast<value_type>(1.0), A, U, V);
        }
        if (iternum % 2 == 1) {
            if (ortho) El::lq::ExplicitUnitary(V);
            base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, V, U);
            El::Copy(U, V);
        }
    }
}

template <typename InputType, typename UType, typename SType, typename VType>
void ApproximateSVD(const InputType &A, UType &U, SType &S, VType &V,
    int rank, base::context_t& context,
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
        std::stringstream err;
        err << "Incompatible matrix dimensions (" << std::min(m, n)
            << ") and target rank (" << rank << ")";
        if (log_lev1)
            params.log_stream << err.str() << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(err.str()));
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
                    static_cast<value_type>(1.0), R, V);
            } else {
                // The above computation, while mathemetically correct for
                // any number of iterations, is not robust enough numerically
                // when number of power iteration is greater than 0.
                El::qr::ExplicitUnitary(Q);
                base::Gemm(El::ADJOINT, El::NORMAL,
                    static_cast<value_type>(1.0), A, Q, V);
            }
        }

        /** Compute factorization & truncate to rank */
        VType B;
        El::SVD(V, V, S, B);
        S.Resize(rank, 1); V.Resize(n, rank);
        VType B1 = base::ColumnView(B, 0, rank);
        base::Gemm(El::NORMAL, El::NORMAL,
            static_cast<value_type>(1.0), Q, B1, U);
    }

    /** Code for m < n */
    if (m < n) {
        int k = std::max(rank, std::min(m,
                params.oversampling_ratio * rank +
                params.oversampling_additive));

        /** Apply sketch transformation on the input matrix */
        VType Q(k, n);
        sketch::JLT_t<InputType, VType> Omega(m, k, context);
        Omega.apply(A, Q, sketch::columnwise_tag());

        /** Power iteration */
        PowerIteration(El::NORMAL, El::ADJOINT, El::NORMAL, A, Q, U,
            params.num_iterations, !params.skip_qr);

        if (params.skip_qr) {
            // We should be able to do the same trick as for m>=n if
            // num_iteration == 0, but the LQ factorization in Elemental
            // does not work for this... (do not know why).
            El::lq::ExplicitUnitary(Q);
            base::Gemm(El::NORMAL, El::ADJOINT, static_cast<value_type>(1.0), A, Q, U);
        }

        /** Compute factorization & truncate to rank */
        UType B;
        El::SVD(U, U, S, B);
        S.Resize(rank, 1); U.Resize(m, rank);
        VType B1 = base::ColumnView(B, 0, rank);
        base::Gemm(El::ADJOINT, El::NORMAL, static_cast<value_type>(1.0), Q, B1, V);
    }
}

template <typename InputType, typename SType, typename VType>
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
        std::stringstream err;
        err << "Matrix is not square (" << base::Height(A) << " x "
            << n << ") -- symmetric matrix required";
        if (log_lev1)
            params.log_stream << err.str() << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(err.str()));
    }

    /**
     * Check if sizes match.
     */
    if (rank > n) {
        std::stringstream err;
        err << "Incompatible matrix dimensions (" << n << ") and target rank ("
            << rank << ").";
        if (log_lev1)
            params.log_stream << err.str() << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(err.str()));
    }

    int k = std::max(rank, std::min(n,
            params.oversampling_ratio * rank +
            params.oversampling_additive));

    VType U(n, k), B(k, k), W(k, k);
    V.Resize(n, k);

    /** Apply sketch transformation on the input matrix */
    // Optimally, we will use JLT_t. But the matrix is potentially only
    // specified in sub/super diagonal, so we explicitly form the random matrix.
    VType Omega;
    base::GaussianMatrix(Omega, n, k, context);
    base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, Omega, V);

    /** Power iteration */
    SymmetricPowerIteration(uplo, El::NORMAL, A, V, params.num_iterations,
        !params.skip_qr);

    /** Schur-Rayleigh-Ritz (with SVD), aka factorize & truncate to rank */
    El::qr::ExplicitUnitary(V);
    base::Symm(El::LEFT, uplo, static_cast<value_type>(1.0), A, V, U);
    base::Gemm(El::ADJOINT, El::NORMAL, static_cast<value_type>(1.0), U, V, B);
    El::HermitianEigCtrl<value_type> ctrl;
    ctrl.tridiagEigCtrl.sort = El::DESCENDING;
    El::HermitianEig(uplo, B, S, W, ctrl);
    S.Resize(rank, 1);
    U.Resize(U.Height(), rank);
    VType W1 = base::ColumnView(W, 0, rank);
    base::Gemm(El::NORMAL, El::NORMAL, static_cast<value_type>(1.0), V, W1, U);
    V.Resize(V.Height(), rank);
    El::Copy(U, V);
}


/*******  ANY variants ********/
void ApproximateSymmetricSVD(El::UpperOrLower uplo,
    const boost::any &A, const boost::any &V, const boost::any &S, int rank,
    base::context_t& context,
    approximate_svd_params_t params = approximate_svd_params_t()) {

#define SKYLARK_SVD_ANY_APPLY_DISPATCH(AT, VT, ST)                      \
    if (V.type() == typeid(VT*) && S.type() == typeid(ST*))  {          \
        if (A.type() == typeid(AT*)) {                                  \
            ApproximateSymmetricSVD(uplo, *boost::any_cast<AT*>(A),     \
                *boost::any_cast<VT*>(V), *boost::any_cast<ST*>(S),     \
                rank, context, params);                                 \
            return;                                                     \
        }                                                               \
        if (A.type() == typeid(const AT*)) {                            \
            ApproximateSymmetricSVD(uplo, *boost::any_cast<const AT*>(A), \
                *boost::any_cast<VT*>(V), *boost::any_cast<ST*>(S),     \
                rank, context, params);                                 \
            return;                                                     \
        }                                                               \
     }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::matrix_t,
        mdtypes::matrix_t, mdtypes::matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mdtypes::dist_matrix_t, mdtypes::dist_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
        mdtypes::shared_matrix_t, mdtypes::shared_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
        mdtypes::root_matrix_t, mdtypes::root_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
        mdtypes::matrix_t, mdtypes::matrix_t);

    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::matrix_t,
        mftypes::matrix_t, mftypes::matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
        mftypes::dist_matrix_t, mftypes::dist_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
        mftypes::shared_matrix_t, mftypes::shared_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
        mftypes::root_matrix_t, mftypes::root_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
        mftypes::matrix_t, mftypes::matrix_t);

#endif

    SKYLARK_THROW_EXCEPTION (
        base::nla_exception()
          << base::error_msg(
           "This combination has not yet been implemented for ApproximateSymmetricSVD"));

#undef SKYLARK_SVD_ANY_APPLY_DISPATCH

}

void ApproximateSVD(const boost::any &A,
    const boost::any &U, const boost::any &S,
    const boost::any &V, int rank, base::context_t& context,
    approximate_svd_params_t params = approximate_svd_params_t()) {

#define SKYLARK_SVD_ANY_APPLY_DISPATCH(AT, UT, VT, ST)                  \
    if (V.type() == typeid(VT*) && S.type() == typeid(ST*))  {          \
        if (A.type() == typeid(AT*)) {                                  \
            ApproximateSVD(*boost::any_cast<AT*>(A),              \
                *boost::any_cast<UT*>(U), *boost::any_cast<ST*>(S),     \
                *boost::any_cast<VT*>(V), rank, context, params);       \
            return;                                                     \
        }                                                               \
        if (A.type() == typeid(const AT*)) {                            \
            ApproximateSVD(*boost::any_cast<const AT*>(A),        \
                *boost::any_cast<UT*>(U), *boost::any_cast<ST*>(S),     \
                *boost::any_cast<VT*>(V), rank, context, params);       \
            return;                                                     \
        }                                                               \
     }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::matrix_t,
        mdtypes::matrix_t, mdtypes::matrix_t, mdtypes::matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mdtypes::dist_matrix_t, mdtypes::dist_matrix_t, mdtypes::dist_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
        mdtypes::shared_matrix_t, mdtypes::shared_matrix_t, mdtypes::shared_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
        mdtypes::root_matrix_t, mdtypes::root_matrix_t, mdtypes::root_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
        mdtypes::matrix_t, mdtypes::matrix_t, mdtypes::matrix_t);

    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::matrix_t,
        mftypes::matrix_t, mftypes::matrix_t, mftypes::matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
        mftypes::dist_matrix_t, mftypes::dist_matrix_t, mftypes::dist_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
        mftypes::shared_matrix_t, mftypes::shared_matrix_t, mftypes::shared_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
        mftypes::root_matrix_t, mftypes::root_matrix_t,  mftypes::root_matrix_t);
    SKYLARK_SVD_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
        mftypes::matrix_t, mftypes::matrix_t, mftypes::matrix_t);

#endif

    SKYLARK_THROW_EXCEPTION (
        base::nla_exception()
          << base::error_msg(
           "This combination has not yet been implemented for ApproximateSVD"));

#undef SKYLARK_SVD_ANY_APPLY_DISPATCH

}


}  // namespace nla
}  // namespace skylark

#endif /** SKYLARK_APPROXIMATE_SVD_HPP */
