#include "boost/property_tree/ptree.hpp"

#include "sketchc.hpp"
#include "../../utility/exception.hpp"
#include "../../utility/sketch_archive.hpp"

#ifdef SKYLARK_HAVE_COMBBLAS
#include "CombBLAS.h"
#include "SpParMat.h"
#include "SpParVec.h"
#include "DenseParVec.h"
#endif

# define STRCMP_TYPE(STR, TYPE) \
    if (std::strcmp(str, #STR) == 0) \
        return TYPE;

static sketchc::matrix_type_t str2matrix_type(char *str) {
    STRCMP_TYPE(Matrix, sketchc::MATRIX);
    STRCMP_TYPE(DistMatrix, sketchc::DIST_MATRIX);
    STRCMP_TYPE(DistMatrix_VC_STAR, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_VR_STAR, sketchc::DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VC, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_STAR_VR, sketchc::DIST_MATRIX_VR_STAR);
    STRCMP_TYPE(DistSparseMatrix,   sketchc::DIST_SPARSE_MATRIX);

    return sketchc::MATRIX_TYPE_ERROR;
}

static sketchc::transform_type_t str2transform_type(char *str) {
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
    STRCMP_TYPE(ExpSemigroupRLT, sketchc::ExpSemigroupRLT);

    return sketchc::TRANSFORM_TYPE_ERROR;
}

// Default data types (the ones we use in Python)
typedef sketch::JLT_data_t<double> JLT_data_t;
typedef sketch::FJLT_data_t<double> FJLT_data_t;
typedef sketch::CT_data_t<double> CT_data_t;
typedef sketch::CWT_data_t<size_t, double> CWT_data_t;
typedef sketch::MMT_data_t<size_t, double> MMT_data_t;
typedef sketch::WZT_data_t<size_t, double> WZT_data_t;
typedef sketch::PPT_data_t<double> PPT_data_t;
typedef sketch::GaussianRFT_data_t<double> GaussianRFT_data_t;
typedef sketch::LaplacianRFT_data_t<double> LaplacianRFT_data_t;
typedef sketch::FastGaussianRFT_data_t<double> FastGaussianRFT_data_t;
typedef sketch::ExpSemigroupRLT_data_t<double> ExpSemigroupRLT_data_t;

// Just for shorter notation
#if SKYLARK_HAVE_ELEMENTAL
typedef elem::Matrix<double> Matrix;
typedef elem::DistMatrix<double> DistMatrix;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> DistMatrix_VR_STAR;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistMatrix_VC_STAR;
typedef elem::DistMatrix<double, elem::STAR, elem::VR> DistMatrix_STAR_VR;
typedef elem::DistMatrix<double, elem::STAR, elem::VC> DistMatrix_STAR_VC;
#endif
#ifdef SKYLARK_HAVE_COMBBLAS
typedef SpDCCols< size_t, double > col_t;
typedef SpParMat< size_t, double, col_t > DistSparseMatrix_t;
#endif


extern "C" {

/** Return string defining what is supported */
SKYLARK_EXTERN_API char *sl_supported_sketch_transforms() {
#define QUOTE(x) #x
#define SKDEF(t,i,o) "(\"" QUOTE(t) "\",\"" QUOTE(i) "\",\"" QUOTE(o) "\") "

    return
#if SKYLARK_HAVE_ELEMENTAL
        SKDEF(JLT, Matrix, Matrix)
        SKDEF(JLT, DistMatrix, Matrix)
        SKDEF(JLT, DistMatrix, DistMatrix)
        SKDEF(JLT, DistMatrix_VR_STAR, Matrix)
        SKDEF(JLT, DistMatrix_VC_STAR, Matrix)
        SKDEF(JLT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(JLT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(JLT, DistMatrix_STAR_VR, Matrix)
        SKDEF(JLT, DistMatrix_STAR_VC, Matrix)
        SKDEF(JLT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(JLT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(CT, Matrix, Matrix)
        SKDEF(CT, DistMatrix, Matrix)
        SKDEF(CT, DistMatrix, DistMatrix)
        SKDEF(CT, DistMatrix_VR_STAR, Matrix)
        SKDEF(CT, DistMatrix_VC_STAR, Matrix)
        SKDEF(CT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(CT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(CT, DistMatrix_STAR_VR, Matrix)
        SKDEF(CT, DistMatrix_STAR_VC, Matrix)
        SKDEF(CT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(CT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(CWT, Matrix, Matrix)
        SKDEF(CWT, DistMatrix, Matrix)
        SKDEF(CWT, DistMatrix_VR_STAR, Matrix)
        SKDEF(CWT, DistMatrix_VC_STAR, Matrix)
        SKDEF(CWT, DistMatrix_STAR_VR, Matrix)
        SKDEF(CWT, DistMatrix_STAR_VC, Matrix)
        SKDEF(MMT, Matrix, Matrix)
        SKDEF(MMT, DistMatrix, Matrix)
        SKDEF(MMT, DistMatrix_VR_STAR, Matrix)
        SKDEF(MMT, DistMatrix_VC_STAR, Matrix)
        SKDEF(MMT, DistMatrix_STAR_VR, Matrix)
        SKDEF(MMT, DistMatrix_STAR_VC, Matrix)
        SKDEF(WZT, Matrix, Matrix)
        SKDEF(WZT, DistMatrix, Matrix)
        SKDEF(WZT, DistMatrix_VR_STAR, Matrix)
        SKDEF(WZT, DistMatrix_VC_STAR, Matrix)
        SKDEF(WZT, DistMatrix_STAR_VR, Matrix)
        SKDEF(WZT, DistMatrix_STAR_VC, Matrix)
        SKDEF(PPT, Matrix, Matrix)
        SKDEF(GaussianRFT, Matrix, Matrix)
        SKDEF(GaussianRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(GaussianRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(LaplacianRFT, Matrix, Matrix)
        SKDEF(LaplacianRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(LaplacianRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(ExpSemigroupRLT, Matrix, Matrix)

#if SKYLARK_HAVE_FFTW
        SKDEF(FJLT, DistMatrix_VR_STAR, Matrix)
        SKDEF(FJLT, DistMatrix_VC_STAR, Matrix)
        SKDEF(FastGaussianRFT, Matrix, Matrix)
#endif

#endif

#ifdef SKYLARK_HAVE_COMBBLAS
        SKDEF(CWT, DistSparseMatrix, DistSparseMatrix)
#endif
       "" ;
}

SKYLARK_EXTERN_API const char* sl_strerror(const int error_code) {
    return skylark_strerror(error_code);
}

SKYLARK_EXTERN_API bool sl_has_elemental() {
#if SKYLARK_HAVE_ELEMENTAL
    return true;
#else
    return false;
#endif
}

SKYLARK_EXTERN_API bool sl_has_combblas() {
#if SKYLARK_HAVE_COMBBLAS
    return true;
#else
    return false;
#endif
}

/** Support for skylark::sketch::context_t. */
SKYLARK_EXTERN_API int sl_create_default_context(int seed,
        sketch::context_t **ctxt) {
    SKYLARK_BEGIN_TRY()
        *ctxt = new sketch::context_t(seed, boost::mpi::communicator());
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

SKYLARK_EXTERN_API int sl_create_context(int seed,
        MPI_Comm comm, sketch::context_t **ctxt) {
    SKYLARK_BEGIN_TRY()
        *ctxt = new sketch::context_t(seed,
            boost::mpi::communicator(comm, boost::mpi::comm_duplicate));
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

SKYLARK_EXTERN_API int sl_free_context(sketch::context_t *ctxt) {
    SKYLARK_BEGIN_TRY()
        delete ctxt;
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

SKYLARK_EXTERN_API int sl_context_rank(sketch::context_t *ctxt, int *rank) {
    SKYLARK_BEGIN_TRY()
        *rank = ctxt->rank;
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

SKYLARK_EXTERN_API int sl_context_size(sketch::context_t *ctxt, int *size) {
    SKYLARK_BEGIN_TRY()
        *size = ctxt->size;
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();
    return 0;
}

/** Transforms */
SKYLARK_EXTERN_API int sl_create_sketch_transform(sketch::context_t *ctxt,
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

    AUTO_NEW_DISPATCH(sketchc::JLT, JLT_data_t);
    AUTO_NEW_DISPATCH(sketchc::FJLT, FJLT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::CT, CT_data_t);
    AUTO_NEW_DISPATCH(sketchc::CWT, CWT_data_t);
    AUTO_NEW_DISPATCH(sketchc::MMT, MMT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::WZT, WZT_data_t)
    AUTO_NEW_DISPATCH_1P(sketchc::GaussianRFT, GaussianRFT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::LaplacianRFT, LaplacianRFT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::ExpSemigroupRLT, ExpSemigroupRLT_data_t);
    AUTO_NEW_DISPATCH_1P(sketchc::FastGaussianRFT, FastGaussianRFT_data_t);

    SKYLARK_BEGIN_TRY()
        if (type == sketchc::PPT)  {
            va_list argp;
            va_start(argp, sketch);
            double q = va_arg(argp, int);
            double c = va_arg(argp, double);
            double g = va_arg(argp, double);
            sketchc::sketch_transform_t *r =
                new sketchc::sketch_transform_t(sketchc::PPT,
                    new PPT_data_t(n, s, q, c, g, *ctxt));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    return 0;
}

SKYLARK_EXTERN_API int sl_load_sketch_transform(sketch::context_t *ctxt,
    char *type_, char *data, sketchc::sketch_transform_t **sketch) {

    sketchc::transform_type_t type = str2transform_type(type_);

    std::stringstream json_data(data);
    boost::property_tree::ptree json;
    boost::property_tree::read_json(json_data, json);

# define AUTO_LOAD_DISPATCH(T, C)                                   \
    SKYLARK_BEGIN_TRY()                                             \
        if (type == T)                                              \
            *sketch = new sketchc::sketch_transform_t(type,         \
                          new C(json, *ctxt));                      \
    SKYLARK_END_TRY()                                               \
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    AUTO_LOAD_DISPATCH(sketchc::JLT, JLT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::CT,  CT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::CWT, CWT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::MMT, MMT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::WZT, WZT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::PPT, PPT_data_t);

    AUTO_LOAD_DISPATCH(sketchc::GaussianRFT,  GaussianRFT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::LaplacianRFT, LaplacianRFT_data_t);

    AUTO_LOAD_DISPATCH(sketchc::ExpSemigroupRLT, ExpSemigroupRLT_data_t);
    AUTO_LOAD_DISPATCH(sketchc::FastGaussianRFT, FastGaussianRFT_data_t);

#if SKYLARK_HAVE_FFTW
    AUTO_LOAD_DISPATCH(sketchc::FJLT, FJLT_data_t);
#endif

    return 0;
}

SKYLARK_EXTERN_API int sl_dump_sketch_transform(sketch::context_t *ctxt,
    char *type_, char *filename, sketchc::sketch_transform_t *sketch) {

    sketchc::transform_type_t type = str2transform_type(type_);

    std::ofstream out(filename);
    boost::property_tree::ptree pt;
    skylark::utility::sketch_archive_t ar;

# define AUTO_DUMP_DISPATCH(T, C)                                   \
    SKYLARK_BEGIN_TRY()                                             \
        if (type == T) {                                            \
            pt << *static_cast<C*>(sketch->transform_obj);          \
            ar << pt;                                               \
            out << ar;                                              \
        }                                                           \
    SKYLARK_END_TRY()                                               \
    SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

    AUTO_DUMP_DISPATCH(sketchc::JLT, JLT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::CT,  CT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::CWT, CWT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::MMT, MMT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::WZT, WZT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::PPT, PPT_data_t);

    AUTO_DUMP_DISPATCH(sketchc::GaussianRFT,  GaussianRFT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::LaplacianRFT, LaplacianRFT_data_t);

    AUTO_DUMP_DISPATCH(sketchc::ExpSemigroupRLT, ExpSemigroupRLT_data_t);
    AUTO_DUMP_DISPATCH(sketchc::FastGaussianRFT, FastGaussianRFT_data_t);

#if SKYLARK_HAVE_FFTW
    AUTO_DUMP_DISPATCH(sketchc::FJLT, FJLT_data_t);
#endif

    out.close();
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

    AUTO_DELETE_DISPATCH(sketchc::JLT, JLT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::FJLT, FJLT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::CT, CT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::CWT, CWT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::MMT, MMT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::WZT, WZT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::PPT, PPT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::GaussianRFT, GaussianRFT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::LaplacianRFT, LaplacianRFT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::ExpSemigroupRLT, ExpSemigroupRLT_data_t);
    AUTO_DELETE_DISPATCH(sketchc::FastGaussianRFT, FastGaussianRFT_data_t);

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

#if SKYLARK_HAVE_ELEMENTAL

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::JLT_t, Matrix, Matrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX, sketchc::MATRIX,
        sketch::JLT_t, DistMatrix, Matrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::JLT_t, DistMatrix, DistMatrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::JLT_t, DistMatrix_VR_STAR, Matrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::JLT_t, DistMatrix_VC_STAR, Matrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VR, Matrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VC, Matrix, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::JLT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::JLT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::JLT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC, JLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::CT_t, Matrix, Matrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX, sketchc::MATRIX,
        sketch::CT_t, DistMatrix, Matrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX, sketchc::DIST_MATRIX,
        sketch::CT_t, DistMatrix, DistMatrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::CT_t, DistMatrix_VR_STAR, Matrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::CT_t, DistMatrix_VC_STAR, Matrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::CT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::CT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::MATRIX,
        sketch::CT_t, DistMatrix_STAR_VR, Matrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::MATRIX,
        sketch::CT_t, DistMatrix_STAR_VC, Matrix, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::DIST_MATRIX_STAR_VR,
        sketch::CT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::DIST_MATRIX_STAR_VC,
        sketch::CT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC, CT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::CWT_t, Matrix, Matrix, CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX, sketchc::MATRIX,
        sketch::CWT_t, DistMatrix, Matrix, CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::CWT_t, DistMatrix_VR_STAR, Matrix, CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::CWT_t, DistMatrix_VC_STAR, Matrix, CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VR, Matrix, CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VC, Matrix, CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::MMT_t, Matrix, Matrix, MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX, sketchc::MATRIX,
        sketch::MMT_t, DistMatrix, Matrix, MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::MMT_t, DistMatrix_VR_STAR, Matrix, MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::MMT_t, DistMatrix_VC_STAR, Matrix, MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VR, Matrix, MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VC, Matrix, MMT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::WZT_t, Matrix, Matrix, WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX, sketchc::MATRIX,
        sketch::WZT_t, DistMatrix, Matrix, WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::WZT_t, DistMatrix_VR_STAR, Matrix, WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::WZT_t, DistMatrix_VC_STAR, Matrix, WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_STAR_VR, sketchc::MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VR, Matrix, WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::WZT,
        sketchc::DIST_MATRIX_STAR_VC, sketchc::MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VC, Matrix, WZT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::PPT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::PPT_t, Matrix, Matrix,
        PPT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::GaussianRFT_t, Matrix, Matrix,
        GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::LaplacianRFT_t, Matrix, Matrix,
        LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::ExpSemigroupRLT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::ExpSemigroupRLT_t, Matrix, Matrix,
        ExpSemigroupRLT_data_t);

#if SKYLARK_HAVE_FFTW

    AUTO_APPLY_DISPATCH(sketchc::FJLT,
        sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
        sketch::FJLT_t, DistMatrix_VR_STAR, Matrix,
        FJLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FJLT,
        sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
        sketch::FJLT_t, DistMatrix_VC_STAR, Matrix,
        FJLT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::FastGaussianRFT,
        sketchc::MATRIX, sketchc::MATRIX,
        sketch::FastGaussianRFT_t, Matrix, Matrix,
        FastGaussianRFT_data_t);

#endif
#ifdef SKYLARK_HAVE_COMBBLAS

    AUTO_APPLY_DISPATCH(sketchc::CWT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
        sketch::CWT_t, DistSparseMatrix_t, DistSparseMatrix_t,
        CWT_data_t);

    AUTO_APPLY_DISPATCH(sketchc::MMT,
        sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
        sketch::MMT_t, DistSparseMatrix_t, DistSparseMatrix_t,
        MMT_data_t);

#endif
#endif

    return 0;
}

SKYLARK_EXTERN_API int sl_wrap_raw_matrix(double *data, int m, int n, void **A)
{
#if SKYLARK_HAVE_ELEMENTAL
    Matrix *tmp = new Matrix();
    tmp->Attach(m, n, data, m);
    *A = tmp;
    return 0;
#else
    return 103;
#endif

}

SKYLARK_EXTERN_API int sl_free_raw_matrix_wrap(void *A_) {
#if SKYLARK_HAVE_ELEMENTAL
    delete static_cast<Matrix *>(A_);
    return 0;
#else
    return 103;
#endif
}

} // extern "C"
