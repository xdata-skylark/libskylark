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

boost::any skylark_void2any(const char *type, void *obj) {

    boost::any anyobj;

    # define STRCMP_CONVERT(STR) \
    if (std::strcmp(type, #STR) == 0) \
        anyobj = static_cast<STR*>(obj);
    
    STRCMP_CONVERT(Matrix);
    STRCMP_CONVERT(SharedMatrix);
    STRCMP_CONVERT(RootMatrix);
    STRCMP_CONVERT(DistMatrix);
    STRCMP_CONVERT(DistMatrix_VC_STAR);
    STRCMP_CONVERT(DistMatrix_VR_STAR);
    STRCMP_CONVERT(DistMatrix_STAR_VC);
    STRCMP_CONVERT(DistMatrix_STAR_VR);
    STRCMP_CONVERT(SparseMatrix);

#ifdef SKYLARK_HAVE_COMBBLAS
    STRCMP_CONVERT(DistSparseMatrix);
#endif

    #undef STRCMP_CONVERT
    
    return anyobj;
}

boost::any skylark_void2any_root(const char *type, void *obj) {

    boost::any anyobj;

# define STRCMP_CONVERT(STR, TYP)         \
    if (std::strcmp(type, #STR) == 0) \
        anyobj = static_cast<TYP*>(obj);
    
    STRCMP_CONVERT(Matrix, Matrix);
    STRCMP_CONVERT(SharedMatrix, ElementalMatrix);
    STRCMP_CONVERT(RootMatrix, ElementalMatrix);
    STRCMP_CONVERT(DistMatrix, ElementalMatrix);
    STRCMP_CONVERT(DistMatrix_VC_STAR, ElementalMatrix);
    STRCMP_CONVERT(DistMatrix_VR_STAR, ElementalMatrix);
    STRCMP_CONVERT(DistMatrix_STAR_VC, ElementalMatrix);
    STRCMP_CONVERT(DistMatrix_STAR_VR, ElementalMatrix);
    STRCMP_CONVERT(SparseMatrix, SparseMatrix);

#ifdef SKYLARK_HAVE_COMBBLAS
    STRCMP_CONVERT(DistSparseMatrix, DistSparseMatrix);
#endif

    #undef STRCMP_CONVERT

    return anyobj;
}

#undef STRCMP_CONVERT
