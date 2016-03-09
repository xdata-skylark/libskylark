#include "matrix_types.hpp"

matrix_type_t str2matrix_type(const char *str) {
    STRCMP_TYPE(Matrix,             MATRIX);
    STRCMP_TYPE(SharedMatrix,       SHARED_MATRIX);
    STRCMP_TYPE(RootMatrix,         ROOT_MATRIX);
    STRCMP_TYPE(DistMatrix,         DIST_MATRIX);
    STRCMP_TYPE(DistMatrix_VC_STAR, DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_VR_STAR, DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VC, DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VR, DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(SparseMatrix,       SPARSE_MATRIX);
    STRCMP_TYPE(DistSparseMatrix,   DIST_SPARSE_MATRIX);

    return MATRIX_TYPE_ERROR;
}
