
#ifndef SKYLARK_RAND_SVD_HPP
#define SKYLARK_RAND_SVD_HPP

#include "../config.h"
#include "../base/exception.hpp"
#include "../base/svd.hpp"
#include "../base/QR.hpp"
#include "../base/Gemm.hpp"
#include "../sketch/capi/sketchc.hpp"


#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif

namespace skylark { namespace nla {


struct rand_svd_params_t {

    int oversampling;
    sketch::c::transform_type_t	transform;
    int num_iterations;
    bool skip_qr;

    rand_svd_params_t(int oversampling, sketch::c::transform_type_t transform = sketch::c::transform_type_t::JLT,
        int num_iterations = 0, bool skip_qr = 0) : oversampling(oversampling), transform(transform),
                                  num_iterations(num_iterations), skip_qr(skip_qr) {};
};


#if SKYLARK_HAVE_ELEMENTAL

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


    // TODO: input matrix should provide Height() and Width()
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
	    std::ostringstream msg;
	    msg << "Incompatible matrix dimensions and target rank\n";
	    SKYLARK_THROW_EXCEPTION(
			    skylark::base::skylark_exception()
			    << skylark::base::error_msg(msg.str()));
    }


    /** Apply sketch transformation on the input matrix */
    UMatrixType Q(input_height, sketch_size);

    typedef typename SketchTransform<InputMatrixType, UMatrixType>::data_type sketch_data_type;
    sketch_data_type sketch_data(input_width, sketch_size, context);
    //typedef typename SketchTransform<InputMatrixType, UMatrixType> sketch_transform_type;
    SketchTransform<InputMatrixType, UMatrixType> sketch_transform(sketch_data);
    sketch_transform.apply(A, Q, skylark::sketch::rowwise_tag());

#if 0
    if (params.transform == sketch::c::transform_type_t::JLT)
	{
	  typedef typename skylark::sketch::JLT_t<InputMatrixType, UMatrixType>::data_type sketch_data_type;
	  sketch_data_type sketch_data(input_width, sketch_size, context);
    	  typedef typename skylark::sketch::JLT_t<InputMatrixType, UMatrixType> sketch_transform_type;
	  sketch_transform_type sketch_transform(sketch_data);
    	  sketch_transform.apply(A, Q, skylark::sketch::rowwise_tag());
	}
    else if (params.transform == sketch::c::transform_type_t::FJLT)
	{
	  typedef typename skylark::sketch::FJLT_t<InputMatrixType, UMatrixType>::data_type sketch_data_type;
	  sketch_data_type sketch_data(input_width, sketch_size, context);
    	  typedef typename skylark::sketch::FJLT_t<InputMatrixType, UMatrixType> sketch_transform_type;
	  sketch_transform_type sketch_transform(sketch_data);
    	  sketch_transform.apply(A, Q, skylark::sketch::rowwise_tag());
	}
    else if (params.transform == sketch::c::transform_type_t::CWT)
	{
	  /*typedef typename skylark::sketch::CWT_t<InputMatrixType, UMatrixType>::data_type sketch_data_type;
	  sketch_data_type sketch_data(input_width, sketch_size, context);
    	  typedef typename skylark::sketch::CWT_t<InputMatrixType, UMatrixType> sketch_transform_type;
	  sketch_transform_type sketch_transform(sketch_data);
    	  sketch_transform.apply(A, Q, skylark::sketch::rowwise_tag());*/
	}
    else
	{
	    std::ostringstream msg;
	    msg << "Unknown sketch transform type\n";
	    SKYLARK_THROW_EXCEPTION(
			    skylark::base::skylark_exception()
			    << skylark::base::error_msg(msg.str()));

	}
#endif

    /** The three steps of the sketched svd approach follow:
     *  - apply sketching
     *  - approximate range of A (find Q)
     *  - SVD
     */


    UMatrixType Y;

    /** Q = QR(Q) */
    skylark::base::qr::Explicit(Q);

    /** q steps of subspace iteration */
    for(int step = 0; step < params.num_iterations; step++) {
	    /** Q = QR(A^T * Q) */
	    skylark::base::Gemm(elem::ADJOINT, elem::NORMAL,
			    double(1), A, Q, Y);
	    skylark::base::qr::Explicit(Y);

	    skylark::base::Gemm(elem::NORMAL, elem::NORMAL,
			    double(1), A,Y, Q);
	    if (!params.skip_qr)
	    	skylark::base::qr::Explicit(Q);
    }


    /** SVD of projected A and then project-back left singular vectors */
    UMatrixType B;
    skylark::base::Gemm(elem::ADJOINT, elem::NORMAL,
		    double(1), Q, A, B);
    skylark::base::SVD(B, SV, V);
    skylark::base::Gemm(elem::NORMAL, elem::NORMAL,
		    double(1), Q, B, U);
}
};

#endif

} } /** namespace skylark::nla */


#endif /** SKYLARK_SKETCHED_SVD_HPP */
