#ifndef SKYLARK_SPARSE_MATRIX_HPP
#define SKYLARK_SPARSE_MATRIX_HPP

#include "matrix_types.hpp"
#include "../base/sparse_dist_matrix.hpp"

struct sparse_matrix_wrapper;

extern "C" {

SKYLARK_EXTERN_API int sl_create_sparse_matrix(char *type_, int N, 
    void **X);

SKYLARK_EXTERN_API int sl_sparse_matrix_height(char *type_, int *height,
    void *X);

SKYLARK_EXTERN_API int sl_sparse_matrix_width(char *type_, int *width,
    void *X);

SKYLARK_EXTERN_API int sl_sparse_matrix_nonzeros(char *type_, int *nonzeros,
    void *X);

}

#endif // SKYLARK_SPARSE_MATRIX_HPP