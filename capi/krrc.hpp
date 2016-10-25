#ifndef SKYLARK_KRRC_HPP
#define SKYLARK_KRRC_HPP

#include "kernelc.hpp" // Defines sl_kernel_t

extern "C" {

SKYLARK_EXTERN_API int sl_kernel_ridge(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *Y_type, void *Y_,
    double lambda,  char *A_type, void *A_,
    char *params_json);
}

#endif // SKYLARK_KRRC_HPP
