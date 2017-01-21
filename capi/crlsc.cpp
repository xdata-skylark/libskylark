#include <El.h>

#include "rlscc.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "matrix_types.hpp"
#include "../ml/ml.hpp"


skylark::base::context_t &dref_context(sl_context_t *ctxt);

extern "C" {

SKYLARK_EXTERN_API int sl_kernel_rlsc(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *L_type, void *L_,
    double lambda,  char *A_type, void *A_,
    El::DistMatrix<El::Int> *rcoding_, char *params_json) {

    skylark::base::direction_t direction =
        direction_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;

    boost::property_tree::ptree json_tree;
    std::stringstream data;
    data << params_json;
    boost::property_tree::read_json(data, json_tree);
    skylark::ml::rlsc_params_t params(data);

    auto *rcoding = new std::vector<El::Int>();

    SKYLARK_BEGIN_TRY()
        skylark::ml::KernelRLSC(direction,
            k->kernel_obj, 
            skylark_void2any(X_type, X_),
            skylark_void2any(L_type, L_),
            lambda, 
            skylark_void2any(A_type, A_),
            *rcoding,
            params);
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    rcoding_->Resize(1, rcoding->size());
    for (int i = 0; i < rcoding->size(); ++i) {
        rcoding_->Set(0, i, rcoding->at(i));
    }

    delete rcoding;
    return 0;
}


SKYLARK_EXTERN_API int sl_approximate_kernel_rlsc(
    int direction_, sl_kernel_t *k,
    char *X_type, void *X_, char *L_type, void *L_,
    double lambda, sketch::sketch_transform_container_t<El::DistMatrix<double>,
        El::DistMatrix<double> > **S_,
    char *W_type, void *W_, El::DistMatrix<El::Int> *rcoding_,
    int s, sl_context_t *ctxt,
    char *params_json) {

    skylark::base::direction_t direction =
        direction_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;

    boost::property_tree::ptree json_tree;
    std::stringstream data;
    data << params_json;
    boost::property_tree::read_json(data, json_tree);
    skylark::ml::rlsc_params_t params(data);
    

    auto *rcoding = new std::vector<El::Int>();
    
    *S_ = new sketch::sketch_transform_container_t<
            El::DistMatrix<double>, El::DistMatrix<double> >();

    SKYLARK_BEGIN_TRY()
        skylark::ml::ApproximateKernelRLSC(direction,
            k->kernel_obj, 
            skylark_void2any(X_type, X_),
            skylark_void2any(L_type, L_),
            lambda,
            **S_,
            skylark_void2any(W_type, W_),
            *rcoding,
            El::Int(s),
            dref_context(ctxt),
            params);
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    rcoding_->Resize(1, rcoding->size());
    for (int i = 0; i < rcoding->size(); ++i) {
        rcoding_->Set(0, i, rcoding->at(i));
    }

    delete rcoding;
    return 0;
}


} // extern "C"
