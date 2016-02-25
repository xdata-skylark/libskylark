#include "matrix_types.hpp"

namespace sketchc = skylark::sketch::c;

sketchc::matrix_type_t str2matrix_type(const char *str) {
    STRCMP_TYPE(Matrix,             sketchc::MATRIX);
    STRCMP_TYPE(SharedMatrix,       sketchc::SHARED_MATRIX);
    STRCMP_TYPE(RootMatrix,         sketchc::ROOT_MATRIX);
    STRCMP_TYPE(DistMatrix,         sketchc::DIST_MATRIX);
    STRCMP_TYPE(DistMatrix_VC_STAR, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_VR_STAR, sketchc::DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VC, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VR, sketchc::DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(SparseMatrix,       sketchc::SPARSE_MATRIX);
    STRCMP_TYPE(DistSparseMatrix,   sketchc::DIST_SPARSE_MATRIX);

    return sketchc::MATRIX_TYPE_ERROR;
}
