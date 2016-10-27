#include <El.h>

#include "krrc.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "matrix_types.hpp"
#include "../ml/ml.hpp"


skylark::base::context_t &dref_context(sl_context_t *ctxt);

extern "C" {

SKYLARK_EXTERN_API int sl_kernel_ridge(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *Y_type, void *Y_,
    double lambda,  char *A_type, void *A_,
    char *params_json) {

    skylark::base::direction_t direction =
        direction_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;

    boost::property_tree::ptree json_tree;
    std::stringstream data;
    data << params_json;
    boost::property_tree::read_json(data, json_tree);
    skylark::ml::krr_params_t params;

    SKYLARK_BEGIN_TRY()
        skylark::ml::KERNELRIDGE(direction,
            k->kernel_obj, 
            skylark_void2any(X_type, X_),
            skylark_void2any(Y_type, Y_),
            lambda, 
            skylark_void2any(A_type, A_), 
            params);
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    return 0;
}

SKYLARK_EXTERN_API int sl_approximate_kernel_ridge(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *Y_type, void *Y_,
    double lambda,
    void *S_,
    char *W_type, void *W_,
    int s,
    sl_context_t *ctxt,
    char *params_json) {

    skylark::base::direction_t direction =
        direction_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;

    boost::property_tree::ptree json_tree;
    std::stringstream data;
    data << params_json;
    boost::property_tree::read_json(data, json_tree);
    skylark::ml::krr_params_t params;

    auto S = static_cast<sketch::sketch_transform_container_t<
        El::DistMatrix<double>, El::DistMatrix<double> > *> (S_);

    SKYLARK_BEGIN_TRY()
        skylark::ml::APPROXIMATEKERNELRIDGE(direction,
            k->kernel_obj, 
            skylark_void2any(X_type, X_),
            skylark_void2any(Y_type, Y_),
            lambda,
            S, 
            skylark_void2any(W_type, W_),
            El::Int(s),
            dref_context(ctxt),
            params);
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    return -0;
}

} // extern "C"