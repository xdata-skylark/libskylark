#ifndef SKYLARK_KERNELC_HPP
#define SKYLARK_KERNELC_HPP

struct sl_kernel_t;

extern "C" {

SKYLARK_EXTERN_API int sl_create_kernel(
       char *type_, int N, sl_kernel_t **kernel, ...);

SKYLARK_EXTERN_API int sl_kernel_gram(int dirX_, int dirY_,
    sl_kernel_t *k,  char *X_type, void *X_, char *Y_type, void *Y_,
    char *K_type, void *K_);
}

#endif // SKYLARK_KERNELC_HPP
