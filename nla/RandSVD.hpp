#ifndef SKYLARK_RAND_SVD_HPP
#define SKYLARK_RAND_SVD_HPP

#include "config.h"
#include "../base/exception.hpp"
#include "../base/svd.hpp"
#include "../base/QR.hpp"
#include "../base/Gemm.hpp"
#include "../sketch/capi/sketchc.hpp"


#include <El.hpp>

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
        base::Gemm(El::NORMAL, El::NORMAL, 1.0, A, V, U);
        if (ortho) El::Scale(1.0 / El::Nrm2(U), U);
    } else {
        if (ortho) base::qr::ExplicitUnitary(V);
        for(int i = 0; i < iternum; i++) {
            base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
            if (ortho) base::qr::ExplicitUnitary(U);
            base::Gemm(adjorientation, El::NORMAL, 1.0, A, U, V);
            if (ortho) base::qr::ExplicitUnitary(V);
        }
        base::Gemm(orientation, El::NORMAL, 1.0, A, V, U);
        if (ortho) base::qr::ExplicitUnitary(U);
    }
}
struct rand_svd_params_t : public base::params_t {

    int oversampling;
    int num_iterations;
    bool skip_qr;

    rand_svd_params_t(int oversampling,
        int num_iterations = 0,
        bool skip_qr = 0,
        bool am_i_printing = 0,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, debug_level),
        oversampling(oversampling),  num_iterations(num_iterations),
        skip_qr(skip_qr) {};
};

template < template <typename, typename> class SketchTransform >
struct randsvd_t {

template <typename InputMatrixType,
          typename UMatrixType,
          typename SingularValuesMatrixType,
          typename VMatrixType>
void operator()(InputMatrixType &A,
    int target_rank,
    UMatrixType &U,
    SingularValuesMatrixType &SV,
    VMatrixType &V,
    rand_svd_params_t params,
    skylark::base::context_t& context) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    int input_height = A.Height();
    int input_width  = A.Width();
    int sketch_size = target_rank + params.oversampling;


    /**
     * Check if sizes match.
     */
    if ((target_rank > std::min(input_height, input_width)) ||
        (sketch_size > input_width) ||
        (sketch_size < target_rank)) {
        std::string msg = "Incompatible matrix dimensions and target rank";
        if (log_lev1)
            params.log_stream << msg << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(msg));
    }

    /** Apply sketch transformation on the input matrix */
    UMatrixType Q(input_height, sketch_size);

    typedef typename SketchTransform<InputMatrixType, UMatrixType>::data_type
        sketch_data_type;
    sketch_data_type sketch_data(input_width, sketch_size, context);
    SketchTransform<InputMatrixType, UMatrixType> sketch_transform(sketch_data);
    sketch_transform.apply(A, Q, sketch::rowwise_tag());

    UMatrixType Y;  // TODO select type
    PowerIteration(El::ADJOINT, A, Q, Y, params.num_iterations, !params.skip_qr);

    UMatrixType B;
    El::Transpose(Y, B);
    base::SVD(B, SV, V);
    base::Gemm(El::NORMAL, El::NORMAL, 1.0, Q, B, U);
}
};

} } /** namespace skylark::nla */


#endif /** SKYLARK_SKETCHED_SVD_HPP */
