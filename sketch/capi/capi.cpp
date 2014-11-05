#include "boost/property_tree/ptree.hpp"

#include "sketchc.hpp"
#include "../../base/exception.hpp"
#include "../../base/sparse_matrix.hpp"

#ifdef SKYLARK_HAVE_COMBBLAS
#include "CombBLAS.h"
#include "SpParMat.h"
#include "SpParVec.h"
#include "DenseParVec.h"
#endif

# define STRCMP_TYPE(STR, TYPE) \
    if (std::strcmp(str, #STR) == 0) \
        return TYPE;

static sketchc::matrix_type_t str2matrix_type(const char *str) {
    STRCMP_TYPE(Matrix,     sketchc::MATRIX);
    STRCMP_TYPE(SharedMatrix,  sketchc::SHARED_MATRIX);
    STRCMP_TYPE(RootMatrix, sketchc::ROOT_MATRIX);
    STRCMP_TYPE(DistMatrix, sketchc::DIST_MATRIX);
    STRCMP_TYPE(DistMatrix_VC_STAR, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_VR_STAR, sketchc::DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VC, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VR, sketchc::DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(SparseMatrix,       sketchc::SPARSE_MATRIX);
    STRCMP_TYPE(DistSparseMatrix,   sketchc::DIST_SPARSE_MATRIX);

    return sketchc::MATRIX_TYPE_ERROR;
}

static sketchc::transform_type_t str2transform_type(const char *str) {
    STRCMP_TYPE(JLT, sketchc::JLT);
    STRCMP_TYPE(CT, sketchc::CT);
    STRCMP_TYPE(FJLT, sketchc::FJLT);
    STRCMP_TYPE(CWT, sketchc::CWT);
    STRCMP_TYPE(MMT, sketchc::MMT);
    STRCMP_TYPE(WZT, sketchc::WZT);
    STRCMP_TYPE(PPT, sketchc::PPT);
    STRCMP_TYPE(GaussianRFT, sketchc::GaussianRFT);
    STRCMP_TYPE(LaplacianRFT, sketchc::LaplacianRFT);
    STRCMP_TYPE(FastGaussianRFT, sketchc::FastGaussianRFT);
    STRCMP_TYPE(FastMaternRFT, sketchc::FastMaternRFT);
    STRCMP_TYPE(ExpSemigroupRLT, sketchc::ExpSemigroupRLT);

    return sketchc::TRANSFORM_TYPE_ERROR;
}

// Just for shorter notation
typedef elem::Matrix<double> Matrix;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> SharedMatrix;
typedef elem::DistMatrix<double, elem::CIRC, elem::CIRC> RootMatrix;
typedef elem::DistMatrix<double> DistMatrix;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> DistMatrix_VR_STAR;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistMatrix_VC_STAR;
typedef elem::DistMatrix<double, elem::STAR, elem::VR> DistMatrix_STAR_VR;
typedef elem::DistMatrix<double, elem::STAR, elem::VC> DistMatrix_STAR_VC;
typedef base::sparse_matrix_t<double> SparseMatrix;
#ifdef SKYLARK_HAVE_COMBBLAS
typedef SpDCCols< size_t, double > col_t;
typedef SpParMat< size_t, double, col_t > DistSparseMatrix;
#endif


extern "C" {

/** Return string defining what is supported */
SKYLARK_EXTERN_API char *sl_supported_sketch_transforms() {
#define QUOTE(x) #x
#define SKDEF(t,i,o) "(\"" QUOTE(t) "\",\"" QUOTE(i) "\",\"" QUOTE(o) "\") "

    return
        SKDEF(JLT, Matrix, Matrix)
        SKDEF(JLT, SparseMatrix, Matrix)
        SKDEF(JLT, DistMatrix, RootMatrix)
        SKDEF(JLT, DistMatrix, SharedMatrix)
        SKDEF(JLT, DistMatrix, DistMatrix)
        SKDEF(JLT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(JLT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(JLT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(JLT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(JLT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(JLT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(JLT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(JLT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(JLT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(JLT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(JLT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(JLT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

        SKDEF(CT, Matrix, Matrix)
        SKDEF(CT, SparseMatrix, Matrix)
        SKDEF(CT, DistMatrix, RootMatrix)
        SKDEF(CT, DistMatrix, SharedMatrix)
        SKDEF(CT, DistMatrix, DistMatrix)
        SKDEF(CT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(CT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(CT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(CT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(CT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(CT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(CT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(CT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(CT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(CT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(CT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(CT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

        SKDEF(CWT, Matrix, Matrix)
        SKDEF(CWT, SparseMatrix, Matrix)
        SKDEF(CWT, SparseMatrix, SparseMatrix)
        SKDEF(CWT, DistMatrix, RootMatrix)
        SKDEF(CWT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(CWT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(CWT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(CWT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(CWT, DistMatrix, SharedMatrix)
        SKDEF(CWT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(CWT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(CWT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(CWT, DistMatrix_STAR_VC, SharedMatrix)

        SKDEF(MMT, Matrix, Matrix)
        SKDEF(MMT, SparseMatrix, Matrix)
        SKDEF(MMT, SparseMatrix, SparseMatrix)
        SKDEF(MMT, DistMatrix, RootMatrix)
        SKDEF(MMT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(MMT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(MMT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(MMT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(MMT, DistMatrix, SharedMatrix)
        SKDEF(MMT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(MMT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(MMT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(MMT, DistMatrix_STAR_VC, SharedMatrix)

        SKDEF(WZT, Matrix, Matrix)
        SKDEF(WZT, SparseMatrix, Matrix)
        SKDEF(WZT, SparseMatrix, SparseMatrix)
        SKDEF(WZT, DistMatrix, RootMatrix)
        SKDEF(WZT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(WZT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(WZT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(WZT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(WZT, DistMatrix, SharedMatrix)
        SKDEF(WZT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(WZT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(WZT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(WZT, DistMatrix_STAR_VC, SharedMatrix)

        SKDEF(PPT, Matrix, Matrix)
        SKDEF(PPT, SparseMatrix, Matrix)

        SKDEF(GaussianRFT, Matrix, Matrix)
        SKDEF(GaussianRFT, SparseMatrix, Matrix)
        SKDEF(GaussianRFT, DistMatrix, RootMatrix)
        SKDEF(GaussianRFT, DistMatrix, SharedMatrix)
        SKDEF(GaussianRFT, DistMatrix, DistMatrix)
        SKDEF(GaussianRFT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(GaussianRFT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(GaussianRFT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(GaussianRFT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(GaussianRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(GaussianRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(GaussianRFT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(GaussianRFT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(GaussianRFT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(GaussianRFT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(GaussianRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(GaussianRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

        SKDEF(LaplacianRFT, Matrix, Matrix)
        SKDEF(LaplacianRFT, SparseMatrix, Matrix)
        SKDEF(LaplacianRFT, DistMatrix, RootMatrix)
        SKDEF(LaplacianRFT, DistMatrix, SharedMatrix)
        SKDEF(LaplacianRFT, DistMatrix, DistMatrix)
        SKDEF(LaplacianRFT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(LaplacianRFT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(LaplacianRFT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(LaplacianRFT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(LaplacianRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(LaplacianRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(LaplacianRFT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(LaplacianRFT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(LaplacianRFT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(LaplacianRFT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(LaplacianRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(LaplacianRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

        SKDEF(ExpSemigroupRLT, Matrix, Matrix)
        SKDEF(ExpSemigroupRLT, SparseMatrix, Matrix)
        SKDEF(ExpSemigroupRLT, DistMatrix, RootMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix, SharedMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix, DistMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(ExpSemigroupRLT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(ExpSemigroupRLT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(ExpSemigroupRLT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

#ifdef SKYLARK_HAVE_COMBBLAS
        SKDEF(CWT, DistSparseMatrix, Matrix)
        SKDEF(CWT, DistSparseMatrix, DistMatrix)
        SKDEF(CWT, DistSparseMatrix, DistMatrix_VC_STAR)
        SKDEF(CWT, DistSparseMatrix, DistMatrix_VR_STAR)
        SKDEF(MMT, DistSparseMatrix, Matrix)
        SKDEF(MMT, DistSparseMatrix, DistMatrix)
        SKDEF(MMT, DistSparseMatrix, DistMatrix_VC_STAR)
        SKDEF(MMT, DistSparseMatrix, DistMatrix_VR_STAR)
#endif

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT
        SKDEF(FJLT, DistMatrix_VR_STAR, Matrix)
        SKDEF(FJLT, DistMatrix_VC_STAR, Matrix)
        SKDEF(FastGaussianRFT, Matrix, Matrix)
        SKDEF(FastGaussianRFT, SparseMatrix, Matrix)
        SKDEF(FastGaussianRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(FastGaussianRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(FastMaternRFT, Matrix, Matrix)
        SKDEF(FastMaternRFT, SparseMatrix, Matrix)
        SKDEF(FastMaternRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(FastMaternRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
#endif


#ifdef SKYLARK_HAVE_COMBBLAS
        SKDEF(CWT, DistSparseMatrix, DistSparseMatrix)
        SKDEF(CWT, DistSparseMatrix, SparseMatrix)
        SKDEF(MMT, DistSparseMatrix, DistSparseMatrix)
        SKDEF(MMT, DistSparseMatrix, SparseMatrix)
#endif

        "";
}

SKYLARK_EXTERN_API const char* sl_strerror(const int error_code) {
    return skylark_strerror(error_code);
}

SKYLARK_EXTERN_API bool sl_has_elemental() {
    return true;
}

SKYLARK_EXTERN_API bool sl_has_combblas() {
#if SKYLARK_HAVE_COMBBLAS
    return true;
#else
    return false;
#endif
}

/** Support for skylark::base::context_t. */
SKYLARK_EXTERN_API int sl_create_default_context(int seed,
        base::context_t **ctxt) {
    SKYLARK_BEGIN_TRY()
        *ctxt = new base::context_t(seed);
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

SKYLARK_EXTERN_API int sl_create_context(int seed,
        MPI_Comm comm, base::context_t **ctxt) {
    SKYLARK_BEGIN_TRY()
        *ctxt = new base::context_t(seed);
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

SKYLARK_EXTERN_API int sl_free_context(base::context_t *ctxt) {
    SKYLARK_BEGIN_TRY()
        delete ctxt;
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

/** Transforms */
SKYLARK_EXTERN_API int sl_create_sketch_transform(base::context_t *ctxt,
    char *type_, int n, int s,
    sketchc::sketch_transform_t **sketch, ...) {

    sketchc::transform_type_t type = str2transform_type(type_);

# define AUTO_NEW_DISPATCH(T, C)                                    \
    SKYLARK_BEGIN_TRY()                                             \
        if (type == T)                                              \
            *sketch = new sketchc::sketch_transform_t(type,         \
                          new C(n, s, *ctxt));                      \
    SKYLARK_END_TRY()                                               \
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

# define AUTO_NEW_DISPATCH_1P(T, C)                                  \
    SKYLARK_BEGIN_TRY()                                              \
        if (type == T)  {                                            \
            va_list argp;                                            \
            va_start(argp, sketch);                                  \
            double p1 = va_arg(argp, double);                        \
            sketchc::sketch_transform_t *r =                         \
                new sketchc::sketch_transform_t(type,                \
                    new C(n, s, p1, *ctxt));                         \
            va_end(argp);                                            \
            *sketch = r;                                             \
        }                                                            \
    SKYLARK_END_TRY()                                                \
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    AUTO_NEW_DISPATCH(sketchc::JLT, sketch::JLT_data_t);
    AUTO_NEW_DISPATCH(sketchc::FJLT, sketch::FJLT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::CT, sketch::CT_data_t);
    AUTO_NEW_DISPATCH(sketchc::CWT, sketch::CWT_data_t);
    AUTO_NEW_DISPATCH(sketchc::MMT, sketch::MMT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::WZT, sketch::WZT_data_t)
    AUTO_NEW_DISPATCH_1P(sketchc::GaussianRFT, sketch::GaussianRFT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::LaplacianRFT, sketch::LaplacianRFT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::ExpSemigroupRLT, sketch::ExpSemigroupRLT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::FastGaussianRFT, sketch::FastGaussianRFT_data_t);

    SKYLARK_BEGIN_TRY()
        if (type == sketchc::FastMaternRFT)  {
            va_list argp;
            va_start(argp, sketch);
            int order = va_arg(argp, int);
            sketchc::sketch_transform_t *r =
                new sketchc::sketch_transform_t(sketchc::FastMaternRFT,
                    new sketch::FastMaternRFT_data_t(n, s, order, *ctxt));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    SKYLARK_BEGIN_TRY()
        if (type == sketchc::PPT)  {
            va_list argp;
            va_start(argp, sketch);
            int q = va_arg(argp, int);
            double c = va_arg(argp, double);
            double g = va_arg(argp, double);
            sketchc::sketch_transform_t *r =
                new sketchc::sketch_transform_t(sketchc::PPT,
                    new sketch::PPT_data_t(n, s, q, c, g, *ctxt));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    return 0;
}

SKYLARK_EXTERN_API int sl_deserialize_sketch_transform(const char *data,
    sketchc::sketch_transform_t **sketch) {

    std::stringstream json_data(data);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json_data, pt);

    sketch::sketch_transform_data_t *sketch_data =
        sketch::sketch_transform_data_t::from_ptree(pt);
    sketchc::transform_type_t type =
        str2transform_type(sketch_data->get_type().c_str());
    *sketch = new sketchc::sketch_transform_t(type, sketch_data);

    return 0;
}

SKYLARK_EXTERN_API int sl_serialize_sketch_transform(
    const sketchc::sketch_transform_t *sketch, char **data) {

    boost::property_tree::ptree pt = sketch->transform_obj->to_ptree();
    std::stringstream json_data;
    boost::property_tree::write_json(json_data, pt);
    *data = new char[json_data.str().length() + 1];
    std::strcpy(*data, json_data.str().c_str());

    return 0;
}

SKYLARK_EXTERN_API
    int sl_free_sketch_transform(sketchc::sketch_transform_t *S) {

    sketchc::transform_type_t type = S->type;

# define AUTO_DELETE_DISPATCH(T, C)                             \
    SKYLARK_BEGIN_TRY()                                         \
        if (type == T)                                          \
            delete static_cast<C *>(S->transform_obj);          \
    SKYLARK_END_TRY()                                           \
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    AUTO_DELETE_DISPATCH(sketchc::JLT, sketch::JLT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::FJLT, sketch::FJLT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::CT, sketch::CT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::CWT, sketch::CWT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::MMT, sketch::MMT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::WZT, sketch::WZT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::PPT, sketch::PPT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::GaussianRFT, sketch::GaussianRFT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::LaplacianRFT, sketch::LaplacianRFT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::ExpSemigroupRLT, sketch::ExpSemigroupRLT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::FastGaussianRFT, sketch::FastGaussianRFT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::FastMaternRFT, sketch::FastMaternRFT_data_t);

    // Now can delete object
    delete S;
    return 0;
}

SKYLARK_EXTERN_API int
    sl_apply_sketch_transform(sketchc::sketch_transform_t *S_,
                              char *input_, void *A_,
                              char *output_, void *SA_, int dim) {

    sketchc::transform_type_t type = S_->type;
    sketchc::matrix_type_t input   = str2matrix_type(input_);
    sketchc::matrix_type_t output  = str2matrix_type(output_);

# define AUTO_APPLY_DISPATCH(T, I, O, C, IT, OT, CD)                     \
    if (type == T && input == I && output == O) {                        \
        C<IT, OT> S(*static_cast<CD*>(S_->transform_obj));               \
        IT &A = * static_cast<IT*>(A_);                                  \
        OT &SA = * static_cast<OT*>(SA_);                                \
                                                                         \
        SKYLARK_BEGIN_TRY()                                              \
            if (dim == SL_COLUMNWISE)                                    \
                S.apply(A, SA, sketch::columnwise_tag());                \
            if (dim == SL_ROWWISE)                                       \
            S.apply(A, SA, sketch::rowwise_tag());                       \
        SKYLARK_END_TRY()                                                \
        SKYLARK_CATCH_AND_RETURN_ERROR_CODE();                           \
    }

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::JLT_t, Matrix, Matrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::JLT_t, SparseMatrix, Matrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::JLT_t, DistMatrix, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::JLT_t, DistMatrix, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::JLT_t, DistMatrix, DistMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_VR_STAR, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_VC_STAR, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VR, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VC, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::JLT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::JLT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::CT_t, Matrix, Matrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::CT_t, SparseMatrix, Matrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::CT_t, DistMatrix, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::CT_t, DistMatrix, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::CT_t, DistMatrix, DistMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::CT_t, DistMatrix_VR_STAR, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::CT_t, DistMatrix_VC_STAR, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::CT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::CT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::CT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::CT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VR, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VC, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::CT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::CT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::CWT_t, Matrix, Matrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::CWT_t, SparseMatrix, Matrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::SPARSE_MATRIX, sketchc::SPARSE_MATRIX,
        sketch::CWT_t, SparseMatrix, SparseMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::CWT_t, DistMatrix, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_VR_STAR, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_VC_STAR, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VR, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VC, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::CWT_t, DistMatrix, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::MMT_t, Matrix, Matrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::MMT_t, SparseMatrix, Matrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::SPARSE_MATRIX, sketchc::SPARSE_MATRIX,
        sketch::MMT_t, SparseMatrix, SparseMatrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::MMT_t, DistMatrix, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_VR_STAR, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_VC_STAR, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VR, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VC, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::MMT_t, DistMatrix, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::WZT_t, Matrix, Matrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::WZT_t, SparseMatrix, Matrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::SPARSE_MATRIX, sketchc::SPARSE_MATRIX,
        sketch::WZT_t, SparseMatrix, SparseMatrix,
        sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::WZT_t, DistMatrix, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_VR_STAR, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_VC_STAR, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VR, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VC, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::WZT_t, DistMatrix, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::PPT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::PPT_t, Matrix, Matrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::PPT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::PPT_t, SparseMatrix, Matrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::GaussianRFT_t, Matrix, Matrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::GaussianRFT_t, SparseMatrix, Matrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::GaussianRFT_t, DistMatrix, DistMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::GaussianRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::GaussianRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::LaplacianRFT_t, Matrix, Matrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::LaplacianRFT_t, SparseMatrix, Matrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, DistMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::ExpSemigroupRLT_t, Matrix, Matrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::ExpSemigroupRLT_t, SparseMatrix, Matrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX, sketchc::ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX, sketchc::SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix, DistMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::ExpSemigroupRLT_data_t);

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT

    AUTO_APPLY_DISPATCH(sketchc::FJLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::FJLT_t, DistMatrix_VR_STAR, Matrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FJLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::FJLT_t, DistMatrix_VC_STAR, Matrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FastGaussianRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::FastGaussianRFT_t, Matrix, Matrix,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FastGaussianRFT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::FastGaussianRFT_t, SparseMatrix, Matrix,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FastMaternRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::FastMaternRFT_t, Matrix, Matrix,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FastMaternRFT,
        sketchc::SPARSE_MATRIX, sketchc::MATRIX,
        sketch::FastMaternRFT_t, SparseMatrix, Matrix,
        sketch::FastMaternRFT_data_t);

#endif

#ifdef SKYLARK_HAVE_COMBBLAS

    // adding a bunch of sp -> sp_sketch -> dense types
    //FIXME: only tested types, */SOMETHING should work as well...
    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::MATRIX,
        sketch::CWT_t, DistSparseMatrix, Matrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_MATRIX,
        sketch::CWT_t, DistSparseMatrix, DistMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_MATRIX_VC_STAR,
        sketch::CWT_t, DistSparseMatrix, DistMatrix_VC_STAR,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_MATRIX_VR_STAR,
        sketch::CWT_t, DistSparseMatrix, DistMatrix_VR_STAR,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::MATRIX,
        sketch::MMT_t, DistSparseMatrix, Matrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_MATRIX,
        sketch::MMT_t, DistSparseMatrix, DistMatrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_MATRIX_VC_STAR,
        sketch::MMT_t, DistSparseMatrix, DistMatrix_VC_STAR,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_MATRIX_VR_STAR,
        sketch::MMT_t, DistSparseMatrix, DistMatrix_VR_STAR,
        sketch::MMT_data_t);
#endif

#ifdef SKYLARK_HAVE_COMBBLAS

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
        sketch::CWT_t, DistSparseMatrix, DistSparseMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::SPARSE_MATRIX,
        sketch::CWT_t, DistSparseMatrix, SparseMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
        sketch::MMT_t, DistSparseMatrix, DistSparseMatrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::SPARSE_MATRIX,
        sketch::MMT_t, DistSparseMatrix, SparseMatrix,
        sketch::MMT_data_t);

#endif

    return 0;
}

SKYLARK_EXTERN_API int sl_wrap_raw_matrix(double *data, int m, int n, void **A)
{
    Matrix *tmp = new Matrix();
    tmp->Attach(m, n, data, m);
    *A = tmp;
    return 0;
}

SKYLARK_EXTERN_API int sl_free_raw_matrix_wrap(void *A_) {
    delete static_cast<Matrix *>(A_);
    return 0;
}


SKYLARK_EXTERN_API int sl_wrap_raw_sp_matrix(int *indptr, int *ind, double *data,
    int nnz, int n_rows, int n_cols, void **A)
{
    SparseMatrix *tmp = new SparseMatrix();
    tmp->attach(indptr, ind, data, nnz, n_rows, n_cols);
    *A = tmp;
    return 0;
}

SKYLARK_EXTERN_API int sl_free_raw_sp_matrix_wrap(void *A_) {
    delete static_cast<SparseMatrix *>(A_);
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_struct_updated(void *A_,
        bool *struct_updated) {
    *struct_updated = static_cast<SparseMatrix *>(A_)->struct_updated();
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_reset_update_flag(void *A_) {
    static_cast<SparseMatrix *>(A_)->reset_update_flag();
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_nnz(void *A_, int *nnz) {
    *nnz = static_cast<SparseMatrix *>(A_)->nonzeros();
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_data(void *A_, int32_t *indptr,
        int32_t *indices, double *values) {
    static_cast<SparseMatrix *>(A_)->detach(indptr, indices, values);
    return 0;
}

} // extern "C"
