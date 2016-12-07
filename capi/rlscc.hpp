#ifndef SKYLARK_RLSCC_HPP
#define SKYLARK_RLSCC_HPP

#include "kernelc.hpp" // Defines sl_kernel_t

extern "C" {

SKYLARK_EXTERN_API int sl_kernel_rlsc(
    int direction_,  sl_kernel_t *k,
    char *X_type, void *X_, char *L_type, void *L_,
    double lambda,  char *A_type, void *A_,
    void *rcoding_,
    char *params_json);
}

#endif // SKYLARK_KRRC_HPP