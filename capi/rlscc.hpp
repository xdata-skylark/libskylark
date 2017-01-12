#ifndef SKYLARK_RLSCC_HPP
#define SKYLARK_RLSCC_HPP

#include "kernelc.hpp" // Defines sl_kernel_t
#include "sketchc.hpp" // Defines sl_sketch_transform_t

extern "C" {

SKYLARK_EXTERN_API int sl_kernel_rlsc(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *L_type, void *L_,
    double lambda,  char *A_type, void *A_,
    El::DistMatrix<El::Int> *rcoding_,
    char *params_json);

SKYLARK_EXTERN_API int sl_approximate_kernel_rlsc(
    int direction_, sl_kernel_t *k,
    char *X_type, void *X_, char *L_type, void *L_,
    double lambda, sketch::sketch_transform_container_t<El::DistMatrix<double>, 
        El::DistMatrix<double> > **S_,
    char *W_type, void *W_, El::DistMatrix<El::Int> *rcoding_,
    int s, sl_context_t *ctxt,
    char *params_json);
}

#endif // SKYLARK_KRRC_HPP
