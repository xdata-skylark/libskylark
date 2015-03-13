#ifndef SKYLARK_RAND_SVD_HPP
#define SKYLARK_RAND_SVD_HPP


#include <El.hpp>
#include "../sketch/sketch.hpp"

namespace skylark { namespace nla {

/**
 * Power iteration from a specific starting vector (the V input).
 */
template<typename MatrixType, typename LeftType, typename RightType>
void PowerIteration(El::Orientation orientation, const MatrixType &A, 
    RightType &V, LeftType &U,
    int iternum, bool ortho = false) {

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef RightType right_type;
    typedef LeftType left_type;

    index_t m = base::Height(A);
    index_t n = base::Width(A);
    index_t k = base::Width(V);

    El::Orientation adjorientation;
    if (orientation == El::ADJOINT || orientation == El::TRANSPOSE) {
        U.Resize(n, k);
        adjorientation = El::NORMAL;
    } else {
        U.Resize(m, k);
        adjorientation = El::ADJOINT;
    }

    if (k == 1) {
        if (ortho) El::Scale(1.0 / El::Nrm2(V), V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
            if (ortho) El::Scale(1.0 / El::Nrm2(U), U);
            base::Gemm(adjorientation, El::NORMAL, 1.0, A, U, V);
            if (ortho) El::Scale(1.0 / El::Nrm2(V), V);
        }
        base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
     } else {
        if (ortho) base::qr::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
            if (ortho) base::qr::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::NORMAL, 1.0, A, U, V);
            if (ortho) base::qr::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
     }
}

struct approximate_svd_params_t : public base::params_t {

    int oversampling_ratio, oversampling_additive;
    int num_iterations;
    bool skip_qr;

    approximate_svd_params_t(int oversampling_ratio = 2,
        int oversampling_additive = 0,
        int num_iterations = 0,
        bool skip_qr = 0,
        bool am_i_printing = 0,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, debug_level),
        oversampling_ratio(oversampling_ratio),  
        oversampling_additive(oversampling_additive),
        num_iterations(num_iterations), skip_qr(skip_qr) {};
};

template <typename InputType, typename UType, typename SType, typename VType>
void ApproximateSVD(InputType &A, UType &U, SType &S, VType &V, int rank,
    base::context_t& context,
    approximate_svd_params_t params = approximate_svd_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    int m = base::Height(A);
    int n = base::Width(A);
    int k = std::max(rank, std::min(n,
            params.oversampling_ratio * rank + params.oversampling_additive));

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

    /** Apply sketch transformation on the input matrix */
    UType Q(m, k);
    sketch::JLT_t<InputType, UType> Omega(n, k, context);
    Omega.apply(A, Q, sketch::rowwise_tag());

    UType Y;  // TODO select type
    PowerIteration(El::ADJOINT, A, Q, Y, params.num_iterations, !params.skip_qr);


    UType B;
    El::Transpose(Y, B);
    El::SVD(B, S, V);
    base::Gemm(El::NORMAL, El::NORMAL, 1.0, Q, B, U);
}

} } /** namespace skylark::nla */


#endif /** SKYLARK_SKETCHED_SVD_HPP */
