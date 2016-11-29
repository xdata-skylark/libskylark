#ifndef SKYLARK_KRRC_HPP
#define SKYLARK_KRRC_HPP

#include "kernelc.hpp" // Defines sl_kernel_t

typedef sketch::sketch_transform_container_t<El::DistMatrix<double>, 
    El::DistMatrix<double> > sketch_transform_container_t_DMD;

extern "C" {

SKYLARK_EXTERN_API int sl_faster_kernel_ridge(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *Y_type, void *Y_,
    double lambda,  char *A_type, void *A_,
    int s,
    sl_context_t *ctxt,
    char *params_json);

}

#endif // SKYLARK_KRRC_HPP