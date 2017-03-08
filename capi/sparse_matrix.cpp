#include "sparse_matrix.hpp"

enum sparse_matrix_type_t {
    SPARSE_MATRIX_TYPE_ERROR,
    SPARSE_MATRIX_T,
    SPARSE_DIST_MATRIX_T
};

static sparse_matrix_type_t str2sparse_matrix_type(const char *str) {
    STRCMP_TYPE(SparseMatrix, SPARSE_MATRIX_T);
    STRCMP_TYPE(SparseDistMatrix, SPARSE_DIST_MATRIX_T);

    return SPARSE_MATRIX_TYPE_ERROR;
}

extern "C" {

SKYLARK_EXTERN_API int sl_create_sparse_matrix(char *type_, int N, 
    void **X) {
    
    sparse_matrix_type_t type = str2sparse_matrix_type(type_);

    if (type == SPARSE_MATRIX_TYPE_ERROR)
        return 111;
    else if (type == SPARSE_MATRIX_T)
        *X = new SparseMatrix;
    else if (type == SPARSE_DIST_MATRIX_T)
        *X = new SparseDistMatrix;

    return 0;
}

SKYLARK_EXTERN_API int sl_sparse_matrix_height(char *type_, int *height,
    void *X) {

    sparse_matrix_type_t type = str2sparse_matrix_type(type_);

    if (type == SPARSE_MATRIX_TYPE_ERROR) {
        return 111;
    }
    else if (type == SPARSE_MATRIX_T) {
        SparseMatrix* sparse_matrix = static_cast<SparseMatrix*>(X);
        *height = sparse_matrix->height();
    }

    return 0;
}

SKYLARK_EXTERN_API int sl_sparse_matrix_width(char *type_, int *width,
    void *X) {
    
    sparse_matrix_type_t type = str2sparse_matrix_type(type_);

    if (type == SPARSE_MATRIX_TYPE_ERROR) {
        return 111;
    }
    else if (type == SPARSE_MATRIX_T) {
        SparseMatrix* sparse_matrix = static_cast<SparseMatrix*>(X);
        *width = sparse_matrix->width();
    }

    return 0;
}

SKYLARK_EXTERN_API int sl_sparse_matrix_nonzeros(char *type_, int *nonzeros,
    void *X) {

    sparse_matrix_type_t type = str2sparse_matrix_type(type_);

    if (type == SPARSE_MATRIX_TYPE_ERROR) {
        return 111;
    }
    else if (type == SPARSE_MATRIX_T) {
        SparseMatrix* sparse_matrix = static_cast<SparseMatrix*>(X);
        *nonzeros = sparse_matrix->nonzeros();
    }

    return 0;
}

} // extern "C"
