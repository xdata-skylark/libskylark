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

struct sparse_matrix_wrapper {
    const sparse_matrix_type_t type;
    std::shared_ptr<SparseMatrix> sparse_matrix;
    std::shared_ptr<SparseDistMatrix> sparse_dist_matrix;

    sparse_matrix_wrapper(sparse_matrix_type_t type_) : type(type_) {
        if (type == SPARSE_MATRIX_T)
            sparse_matrix = std::shared_ptr<SparseMatrix> (new SparseMatrix);
        //else if (type == SPARSE_DIST_MATRIX_T)
            //sparse_dist_matrix = std::shared_ptr<SparseDistMatrix> 
            //    (new skylark::base::sparse_vc_star_matrix_t<double> 
            //        (0, 0, El::Grid());
    }

};

extern "C" {

SKYLARK_EXTERN_API int sl_create_sparse_matrix(char *type_, int N, 
    sparse_matrix_wrapper **sparse_matrix) {
    
    sparse_matrix_type_t type = str2sparse_matrix_type(type_);

    if (type == SPARSE_MATRIX_TYPE_ERROR)
        return 111;
    else
        *sparse_matrix = new sparse_matrix_wrapper(type);

    return 0;
}

} // extern "C"
