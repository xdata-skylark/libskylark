#include <El.h>

#include "krrc.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "matrix_types.hpp"
#include "../ml/ml.hpp"

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
            skylark_void2any_root(X_type, X_),
            skylark_void2any_root(Y_type, Y_),
            lambda, 
            skylark_void2any_root(A_type, A_), 
            params);
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    return 0;
}

} // extern "C"