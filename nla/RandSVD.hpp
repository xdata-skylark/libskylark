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
     * Sanity checks, raise an exception if:
     *   i)   the target rank is too large for the given input matrix or
     *   ii)  the number of columns of the sketched matrix either:
     *        - exceeds its width or
     *        - is less than the target rank
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

    /** The three steps of the sketched svd approach follow:
     *  - apply sketching
     *  - approximate range of A (find Q)
     *  - SVD
     */
    UMatrixType Y;

    /** Q = QR(Q) */
    base::qr::ExplicitUnitary(Q);

    /** q steps of subspace iteration */
    for(int step = 0; step < params.num_iterations; step++) {
        /** Q = QR(A^T * Q) */
        base::Gemm(El::ADJOINT, El::NORMAL, double(1), A, Q, Y);
        base::qr::ExplicitUnitary(Y);
        base::Gemm(El::NORMAL, El::NORMAL, double(1), A,Y, Q);
        if (!params.skip_qr)
            base::qr::ExplicitUnitary(Q);
    }

    /** SVD of projected A and then project-back left singular vectors */
    UMatrixType B;
    base::Gemm(El::ADJOINT, El::NORMAL, double(1), Q, A, B);
    base::SVD(B, SV, V);
    base::Gemm(El::NORMAL, El::NORMAL, double(1), Q, B, U);
}
};

} } /** namespace skylark::nla */


#endif /** SKYLARK_SKETCHED_SVD_HPP */
