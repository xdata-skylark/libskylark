#include "boost/property_tree/ptree.hpp"

#include "matrix_types.hpp"
#include "sketchc.hpp"
#include "../base/context.hpp"
#include "../base/exception.hpp"
#include "../sketch/sketch.hpp"

#ifdef SKYLARK_HAVE_COMBBLAS
#include "CombBLAS.h"
#include "SpParMat.h"
#include "SpParVec.h"
#include "DenseParVec.h"
#endif

namespace base = skylark::base;
namespace sketch = skylark::sketch;

enum transform_type_t {
    TRANSFORM_TYPE_ERROR,
    JLT,
    CT,
    FJLT,
    CWT,
    MMT,
    WZT,
    PPT,
    GaussianRFT,
    LaplacianRFT,
    MaternRFT,
    GaussianQRFT,
    LaplacianQRFT,
    FastGaussianRFT,
    FastMaternRFT,
    ExpSemigroupRLT,
    ExpSemigroupQRLT
};

struct sl_sketch_transform_t {
    const transform_type_t type;
    sketch::sketch_transform_data_t * const transform_obj;

    sl_sketch_transform_t(transform_type_t type,
        sketch::sketch_transform_data_t *transform_obj)
        : type(type), transform_obj(transform_obj) {}
};

static transform_type_t str2transform_type(const char *str) {
    STRCMP_TYPE(JLT, JLT);
    STRCMP_TYPE(CT, CT);
    STRCMP_TYPE(FJLT, FJLT);
    STRCMP_TYPE(CWT, CWT);
    STRCMP_TYPE(MMT, MMT);
    STRCMP_TYPE(WZT, WZT);
    STRCMP_TYPE(PPT, PPT);
    STRCMP_TYPE(GaussianRFT, GaussianRFT);
    STRCMP_TYPE(LaplacianRFT, LaplacianRFT);
    STRCMP_TYPE(MaternRFT, MaternRFT);
    STRCMP_TYPE(GaussianQRFT, GaussianQRFT);
    STRCMP_TYPE(LaplacianQRFT, LaplacianQRFT);
    STRCMP_TYPE(FastGaussianRFT, FastGaussianRFT);
    STRCMP_TYPE(FastMaternRFT, FastMaternRFT);
    STRCMP_TYPE(ExpSemigroupRLT, ExpSemigroupRLT);
    STRCMP_TYPE(ExpSemigroupQRLT, ExpSemigroupQRLT);

    return TRANSFORM_TYPE_ERROR;
}

skylark::base::context_t &dref_context(sl_context_t *ctxt);

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
        SKDEF(JLT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(JLT, DistMatrix_VC_STAR, DistMatrix)

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
        SKDEF(CT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(CT, DistMatrix_VC_STAR, DistMatrix)

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
        SKDEF(CWT, DistMatrix, DistMatrix)
        SKDEF(CWT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(CWT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(CWT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(CWT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

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
        SKDEF(MMT, DistMatrix, DistMatrix)
        SKDEF(MMT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(MMT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(MMT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(MMT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

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
        SKDEF(WZT, DistMatrix, DistMatrix)
        SKDEF(WZT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(WZT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(WZT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(WZT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)

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
        SKDEF(GaussianRFT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(GaussianRFT, DistMatrix_VC_STAR, DistMatrix)

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
        SKDEF(LaplacianRFT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(LaplacianRFT, DistMatrix_VC_STAR, DistMatrix)

        SKDEF(MaternRFT, Matrix, Matrix)
        SKDEF(MaternRFT, SparseMatrix, Matrix)
        SKDEF(MaternRFT, DistMatrix, RootMatrix)
        SKDEF(MaternRFT, DistMatrix, SharedMatrix)
        SKDEF(MaternRFT, DistMatrix, DistMatrix)
        SKDEF(MaternRFT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(MaternRFT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(MaternRFT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(MaternRFT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(MaternRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(MaternRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(MaternRFT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(MaternRFT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(MaternRFT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(MaternRFT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(MaternRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(MaternRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(MaternRFT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(MaternRFT, DistMatrix_VC_STAR, DistMatrix)

        SKDEF(GaussianQRFT, Matrix, Matrix)
        SKDEF(GaussianQRFT, SparseMatrix, Matrix)
        SKDEF(GaussianQRFT, DistMatrix, RootMatrix)
        SKDEF(GaussianQRFT, DistMatrix, SharedMatrix)
        SKDEF(GaussianQRFT, DistMatrix, DistMatrix)
        SKDEF(GaussianQRFT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(GaussianQRFT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(GaussianQRFT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(GaussianQRFT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(GaussianQRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(GaussianQRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(GaussianQRFT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(GaussianQRFT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(GaussianQRFT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(GaussianQRFT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(GaussianQRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(GaussianQRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(GaussianQRFT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(GaussianQRFT, DistMatrix_VC_STAR, DistMatrix)

        SKDEF(LaplacianQRFT, Matrix, Matrix)
        SKDEF(LaplacianQRFT, SparseMatrix, Matrix)
        SKDEF(LaplacianQRFT, DistMatrix, RootMatrix)
        SKDEF(LaplacianQRFT, DistMatrix, SharedMatrix)
        SKDEF(LaplacianQRFT, DistMatrix, DistMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(LaplacianQRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(LaplacianQRFT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(LaplacianQRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(LaplacianQRFT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(LaplacianQRFT, DistMatrix_VC_STAR, DistMatrix)

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
        SKDEF(ExpSemigroupRLT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(ExpSemigroupRLT, DistMatrix_VC_STAR, DistMatrix)

        SKDEF(ExpSemigroupQRLT, Matrix, Matrix)
        SKDEF(ExpSemigroupQRLT, SparseMatrix, Matrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix, RootMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix, SharedMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix, DistMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(ExpSemigroupQRLT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
        SKDEF(ExpSemigroupQRLT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VR_STAR, DistMatrix)
        SKDEF(ExpSemigroupQRLT, DistMatrix_VC_STAR, DistMatrix)

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
        SKDEF(FJLT, DistMatrix_VR_STAR, RootMatrix)
        SKDEF(FJLT, DistMatrix_VC_STAR, RootMatrix)
        SKDEF(FJLT, DistMatrix_STAR_VR, RootMatrix)
        SKDEF(FJLT, DistMatrix_STAR_VC, RootMatrix)
        SKDEF(FJLT, DistMatrix_VR_STAR, SharedMatrix)
        SKDEF(FJLT, DistMatrix_VC_STAR, SharedMatrix)
        SKDEF(FJLT, DistMatrix_STAR_VR, SharedMatrix)
        SKDEF(FJLT, DistMatrix_STAR_VC, SharedMatrix)
        SKDEF(FJLT, DistMatrix, DistMatrix)

        SKDEF(FastGaussianRFT, Matrix, Matrix)
        SKDEF(FastGaussianRFT, SparseMatrix, Matrix)
        SKDEF(FastGaussianRFT, SharedMatrix, SharedMatrix)
        SKDEF(FastGaussianRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(FastGaussianRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(FastGaussianRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(FastGaussianRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)

        SKDEF(FastMaternRFT, Matrix, Matrix)
        SKDEF(FastMaternRFT, SparseMatrix, Matrix)
        SKDEF(FastMaternRFT, SharedMatrix, SharedMatrix)
        SKDEF(FastMaternRFT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(FastMaternRFT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(FastMaternRFT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(FastMaternRFT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
#endif

#if SKYLARK_HAVE_FFTW
        SKDEF(PPT, Matrix, Matrix)
        SKDEF(PPT, SparseMatrix, Matrix)
        SKDEF(PPT, SharedMatrix, SharedMatrix)
        SKDEF(PPT, DistMatrix, DistMatrix)
        SKDEF(PPT, DistMatrix_VC_STAR, DistMatrix_VC_STAR)
        SKDEF(PPT, DistMatrix_VR_STAR, DistMatrix_VR_STAR)
        SKDEF(PPT, DistMatrix_STAR_VC, DistMatrix_STAR_VC)
        SKDEF(PPT, DistMatrix_STAR_VR, DistMatrix_STAR_VR)
#endif


#ifdef SKYLARK_HAVE_COMBBLAS
        SKDEF(CWT, DistSparseMatrix, DistSparseMatrix)
        SKDEF(CWT, DistSparseMatrix, SparseMatrix)
        SKDEF(MMT, DistSparseMatrix, DistSparseMatrix)
        SKDEF(MMT, DistSparseMatrix, SparseMatrix)
#endif

        "";
}

/* Transforms */
SKYLARK_EXTERN_API int sl_create_sketch_transform(sl_context_t *ctxt,
    char *type_, int n, int s,
    sl_sketch_transform_t **sketch, ...) {

    transform_type_t type = str2transform_type(type_);

    if (type == TRANSFORM_TYPE_ERROR)
        return 111;
    
# define AUTO_NEW_DISPATCH(T, C)                                    \
    SKYLARK_BEGIN_TRY()                                             \
        if (type == T)                                              \
            *sketch = new sl_sketch_transform_t(type,               \
                new C(n, s, dref_context(ctxt)));                   \
    SKYLARK_END_TRY()                                               \
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

# define AUTO_NEW_DISPATCH_1P(T, C)                                  \
    SKYLARK_BEGIN_TRY()                                              \
        if (type == T)  {                                            \
            va_list argp;                                            \
            va_start(argp, sketch);                                  \
            double p1 = va_arg(argp, double);                        \
            sl_sketch_transform_t *r =                               \
                new sl_sketch_transform_t(type,                      \
                    new C(n, s, p1, dref_context(ctxt)));            \
            va_end(argp);                                            \
            *sketch = r;                                             \
        }                                                            \
    SKYLARK_END_TRY()                                                \
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    AUTO_NEW_DISPATCH(JLT, sketch::JLT_data_t);
    AUTO_NEW_DISPATCH(FJLT, sketch::FJLT_data_t);
    AUTO_NEW_DISPATCH_1P(CT, sketch::CT_data_t);
    AUTO_NEW_DISPATCH(CWT, sketch::CWT_data_t);
    AUTO_NEW_DISPATCH(MMT, sketch::MMT_data_t);
    AUTO_NEW_DISPATCH_1P(WZT, sketch::WZT_data_t)
    AUTO_NEW_DISPATCH_1P(GaussianRFT, sketch::GaussianRFT_data_t);
    AUTO_NEW_DISPATCH_1P(LaplacianRFT, sketch::LaplacianRFT_data_t);

    SKYLARK_BEGIN_TRY()
        if (type == MaternRFT)  {
            va_list argp;
            va_start(argp, sketch);
            double nu = va_arg(argp, double);
            double l = va_arg(argp, double);
            sl_sketch_transform_t *r =
                new sl_sketch_transform_t(MaternRFT,
                    new sketch::
                    MaternRFT_data_t(n, s, nu, l, dref_context(ctxt)));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    SKYLARK_BEGIN_TRY()
        if (type == GaussianQRFT)  {
            va_list argp;
            va_start(argp, sketch);
            double sigma = va_arg(argp, double);
            int skip = va_arg(argp, int);
            int seqdim =
                sketch::
                GaussianQRFT_data_t<base::leaped_halton_sequence_t>::
                qmc_sequence_dim(n);
            base::leaped_halton_sequence_t<double> sequence(seqdim);
            sl_sketch_transform_t *r =
                new sl_sketch_transform_t(GaussianQRFT,
                    new sketch::
                    GaussianQRFT_data_t<base::
                    leaped_halton_sequence_t>(n, s, sigma, sequence, skip,
                        dref_context(ctxt)));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    SKYLARK_BEGIN_TRY()
        if (type == LaplacianQRFT)  {
            va_list argp;
            va_start(argp, sketch);
            double sigma = va_arg(argp, double);
            int skip = va_arg(argp, int);
            int seqdim =
                sketch::
                GaussianQRFT_data_t<base::leaped_halton_sequence_t>::
                qmc_sequence_dim(n);
            base::leaped_halton_sequence_t<double> sequence(seqdim);
            sl_sketch_transform_t *r =
                new sl_sketch_transform_t(GaussianQRFT,
                    new sketch::
                    LaplacianQRFT_data_t<base::
                    leaped_halton_sequence_t>(n, s, sigma, sequence, skip,
                        dref_context(ctxt)));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    AUTO_NEW_DISPATCH_1P(ExpSemigroupRLT,
        sketch::ExpSemigroupRLT_data_t);

    SKYLARK_BEGIN_TRY()
        if (type == ExpSemigroupQRLT)  {
            va_list argp;
            va_start(argp, sketch);
            double beta = va_arg(argp, double);
            int skip = va_arg(argp, int);
            int seqdim =
                sketch::
                ExpSemigroupQRLT_data_t<base::leaped_halton_sequence_t>::
                qmc_sequence_dim(n);
            base::leaped_halton_sequence_t<double> sequence(seqdim);
            sl_sketch_transform_t *r =
                new sl_sketch_transform_t(ExpSemigroupQRLT,
                    new sketch::
                    ExpSemigroupQRLT_data_t<base::
                    leaped_halton_sequence_t>(n, s, beta, sequence, skip,
                        dref_context(ctxt)));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    AUTO_NEW_DISPATCH_1P(FastGaussianRFT,
        sketch::FastGaussianRFT_data_t);

    SKYLARK_BEGIN_TRY()
        if (type == FastMaternRFT)  {
            va_list argp;
            va_start(argp, sketch);
            double nu = va_arg(argp, double);
            double l = va_arg(argp, double);
            sl_sketch_transform_t *r =
                new sl_sketch_transform_t(FastMaternRFT,
                    new sketch::FastMaternRFT_data_t(n, s, nu, l,
                        dref_context(ctxt)));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    SKYLARK_BEGIN_TRY()
        if (type == PPT)  {
            va_list argp;
            va_start(argp, sketch);
            int q = va_arg(argp, int);
            double c = va_arg(argp, double);
            double g = va_arg(argp, double);
            sl_sketch_transform_t *r =
                new sl_sketch_transform_t(PPT,
                    new sketch::PPT_data_t(n, s, q, c, g,
                        dref_context(ctxt)));
            va_end(argp);
            *sketch = r;
        }
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    return 0;
}

SKYLARK_EXTERN_API int sl_deserialize_sketch_transform(const char *data,
    sl_sketch_transform_t **sketch) {

    std::stringstream json_data(data);
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(json_data, pt);

    sketch::sketch_transform_data_t *sketch_data =
        sketch::sketch_transform_data_t::from_ptree(pt);
    transform_type_t type =
        str2transform_type(sketch_data->get_type().c_str());
    *sketch = new sl_sketch_transform_t(type, sketch_data);

    return 0;
}

SKYLARK_EXTERN_API int sl_serialize_sketch_transform(
    const sl_sketch_transform_t *sketch, char **data) {

    boost::property_tree::ptree pt = sketch->transform_obj->to_ptree();
    std::stringstream json_data;
    boost::property_tree::write_json(json_data, pt);
    *data = new char[json_data.str().length() + 1];
    std::strcpy(*data, json_data.str().c_str());

    return 0;
}

SKYLARK_EXTERN_API
    int sl_free_sketch_transform(sl_sketch_transform_t *S) {

    transform_type_t type = S->type;

# define AUTO_DELETE_DISPATCH(T, C)                             \
    SKYLARK_BEGIN_TRY()                                         \
        if (type == T)                                          \
            delete static_cast<C *>(S->transform_obj);          \
    SKYLARK_END_TRY()                                           \
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    AUTO_DELETE_DISPATCH(JLT, sketch::JLT_data_t);
    AUTO_DELETE_DISPATCH(FJLT, sketch::FJLT_data_t);
    AUTO_DELETE_DISPATCH(CT, sketch::CT_data_t);
    AUTO_DELETE_DISPATCH(CWT, sketch::CWT_data_t);
    AUTO_DELETE_DISPATCH(MMT, sketch::MMT_data_t);
    AUTO_DELETE_DISPATCH(WZT, sketch::WZT_data_t);
    AUTO_DELETE_DISPATCH(PPT, sketch::PPT_data_t);
    AUTO_DELETE_DISPATCH(GaussianRFT, sketch::GaussianRFT_data_t);
    AUTO_DELETE_DISPATCH(LaplacianRFT, sketch::LaplacianRFT_data_t);
    AUTO_DELETE_DISPATCH(MaternRFT, sketch::MaternRFT_data_t);
    AUTO_DELETE_DISPATCH(GaussianQRFT,
        sketch::GaussianQRFT_data_t<base::leaped_halton_sequence_t>);
    AUTO_DELETE_DISPATCH(LaplacianQRFT,
        sketch::LaplacianQRFT_data_t<base::leaped_halton_sequence_t>);
    AUTO_DELETE_DISPATCH(ExpSemigroupRLT,
        sketch::ExpSemigroupRLT_data_t);
    AUTO_DELETE_DISPATCH(ExpSemigroupQRLT,
        sketch::ExpSemigroupQRLT_data_t<base::leaped_halton_sequence_t>);
    AUTO_DELETE_DISPATCH(FastGaussianRFT,
        sketch::FastGaussianRFT_data_t);
    AUTO_DELETE_DISPATCH(FastMaternRFT, sketch::FastMaternRFT_data_t);

    // Now can delete object
    delete S;
    return 0;
}

SKYLARK_EXTERN_API int
    sl_apply_sketch_transform(sl_sketch_transform_t *S_,
                              char *input_, void *A_,
                              char *output_, void *SA_, int dim) {

    transform_type_t type = S_->type;
    matrix_type_t input   = str2matrix_type(input_);
    matrix_type_t output  = str2matrix_type(output_);

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
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);         \
    }

    AUTO_APPLY_DISPATCH(JLT,
        MATRIX, MATRIX,
        sketch::JLT_t, Matrix, Matrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        SPARSE_MATRIX, MATRIX,
        sketch::JLT_t, SparseMatrix, Matrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::JLT_t, DistMatrix, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::JLT_t, DistMatrix, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::JLT_t, DistMatrix, DistMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_VR_STAR, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_VC_STAR, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VR, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VC, RootMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::JLT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::JLT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::JLT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(JLT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix, sketch::JLT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        MATRIX, MATRIX,
        sketch::CT_t, Matrix, Matrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        SPARSE_MATRIX, MATRIX,
        sketch::CT_t, SparseMatrix, Matrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::CT_t, DistMatrix, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::CT_t, DistMatrix, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::CT_t, DistMatrix, DistMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::CT_t, DistMatrix_VR_STAR, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::CT_t, DistMatrix_VC_STAR, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::CT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::CT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::CT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::CT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VR, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VC, RootMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::CT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::CT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::CT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::CT_t, DistMatrix_VR_STAR, DistMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::CT_t, DistMatrix_VC_STAR, DistMatrix, sketch::CT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        MATRIX, MATRIX,
        sketch::CWT_t, Matrix, Matrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        SPARSE_MATRIX, MATRIX,
        sketch::CWT_t, SparseMatrix, Matrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        SPARSE_MATRIX, SPARSE_MATRIX,
        sketch::CWT_t, SparseMatrix, SparseMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::CWT_t, DistMatrix, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_VR_STAR, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_VC_STAR, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VR, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VC, RootMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::CWT_t, DistMatrix, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::CWT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        MATRIX, MATRIX,
        sketch::MMT_t, Matrix, Matrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        SPARSE_MATRIX, MATRIX,
        sketch::MMT_t, SparseMatrix, Matrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        SPARSE_MATRIX, SPARSE_MATRIX,
        sketch::MMT_t, SparseMatrix, SparseMatrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::MMT_t, DistMatrix, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_VR_STAR, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_VC_STAR, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VR, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VC, RootMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::MMT_t, DistMatrix, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::MMT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        MATRIX, MATRIX,
        sketch::WZT_t, Matrix, Matrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        SPARSE_MATRIX, MATRIX,
        sketch::WZT_t, SparseMatrix, Matrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        SPARSE_MATRIX, SPARSE_MATRIX,
        sketch::WZT_t, SparseMatrix, SparseMatrix,
        sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::WZT_t, DistMatrix, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_VR_STAR, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_VC_STAR, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VR, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VC, RootMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::WZT_t, DistMatrix, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_VR_STAR, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_VC_STAR, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VR, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(WZT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::WZT_t, DistMatrix_STAR_VC, SharedMatrix, sketch::WZT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        MATRIX, MATRIX,
        sketch::GaussianRFT_t, Matrix, Matrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::GaussianRFT_t, SparseMatrix, Matrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::GaussianRFT_t, DistMatrix, DistMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::GaussianRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::GaussianRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(GaussianRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::GaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        MATRIX, MATRIX,
        sketch::LaplacianRFT_t, Matrix, Matrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::LaplacianRFT_t, SparseMatrix, Matrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, DistMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        MATRIX, MATRIX,
        sketch::LaplacianRFT_t, Matrix, Matrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::LaplacianRFT_t, SparseMatrix, Matrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix, DistMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::LaplacianRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(LaplacianRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::LaplacianRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        MATRIX, MATRIX,
        sketch::MaternRFT_t, Matrix, Matrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::MaternRFT_t, SparseMatrix, Matrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::MaternRFT_t, DistMatrix, RootMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::MaternRFT_t, DistMatrix, SharedMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::MaternRFT_t, DistMatrix, DistMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::MaternRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::MaternRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::MaternRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::MaternRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::MaternRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::MaternRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::MaternRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::MaternRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::MaternRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::MaternRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::MaternRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::MaternRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::MaternRFT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::MaternRFT_data_t);

    AUTO_APPLY_DISPATCH(MaternRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::MaternRFT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::MaternRFT_data_t);

# define AUTO_APPLY_DISPATCH_QUASI(T, I, O, C, IT, OT, CD)               \
    if (type == T && input == I && output == O) {                        \
        C<IT, OT, base::leaped_halton_sequence_t>            \
            S(*static_cast<CD<base::leaped_halton_sequence_t>*>(S_->transform_obj)); \
        IT &A = * static_cast<IT*>(A_);                                  \
        OT &SA = * static_cast<OT*>(SA_);                                \
                                                                         \
        SKYLARK_BEGIN_TRY()                                              \
            if (dim == SL_COLUMNWISE)                                    \
                S.apply(A, SA, sketch::columnwise_tag());                \
            if (dim == SL_ROWWISE)                                       \
            S.apply(A, SA, sketch::rowwise_tag());                       \
        SKYLARK_END_TRY()                                                \
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);         \
    }

   AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        MATRIX, MATRIX,
        sketch::GaussianQRFT_t, Matrix, Matrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::GaussianQRFT_t, SparseMatrix, Matrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix, RootMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix, SharedMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix, DistMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::GaussianQRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::GaussianQRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::GaussianQRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::GaussianQRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(GaussianQRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::GaussianQRFT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::GaussianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        MATRIX, MATRIX,
        sketch::LaplacianQRFT_t, Matrix, Matrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::LaplacianQRFT_t, SparseMatrix, Matrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix, RootMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix, SharedMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix, DistMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::LaplacianQRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::LaplacianQRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::LaplacianQRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::LaplacianQRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(LaplacianQRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::LaplacianQRFT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::LaplacianQRFT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        MATRIX, MATRIX,
        sketch::ExpSemigroupRLT_t, Matrix, Matrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        SPARSE_MATRIX, MATRIX,
        sketch::ExpSemigroupRLT_t, SparseMatrix, Matrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix, DistMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::ExpSemigroupRLT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::ExpSemigroupRLT_data_t);

    AUTO_APPLY_DISPATCH(ExpSemigroupRLT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::ExpSemigroupRLT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::ExpSemigroupRLT_data_t);

   AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        MATRIX, MATRIX,
        sketch::ExpSemigroupQRLT_t, Matrix, Matrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        SPARSE_MATRIX, MATRIX,
        sketch::ExpSemigroupQRLT_t, SparseMatrix, Matrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX, ROOT_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix, RootMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX, SHARED_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix, SharedMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix, DistMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::ExpSemigroupQRLT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::ExpSemigroupQRLT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VR_STAR, DistMatrix,
        sketch::ExpSemigroupQRLT_data_t);

    AUTO_APPLY_DISPATCH_QUASI(ExpSemigroupQRLT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX,
        sketch::ExpSemigroupQRLT_t, DistMatrix_VC_STAR, DistMatrix,
        sketch::ExpSemigroupQRLT_data_t);

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_VR_STAR, ROOT_MATRIX,
        sketch::FJLT_t, DistMatrix_VR_STAR, RootMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_VC_STAR, ROOT_MATRIX,
        sketch::FJLT_t, DistMatrix_VC_STAR, RootMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_STAR_VR, ROOT_MATRIX,
        sketch::FJLT_t, DistMatrix_STAR_VR, RootMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_STAR_VC, ROOT_MATRIX,
        sketch::FJLT_t, DistMatrix_STAR_VC, RootMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_VR_STAR, SHARED_MATRIX,
        sketch::FJLT_t, DistMatrix_VR_STAR, SharedMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_VC_STAR, SHARED_MATRIX,
        sketch::FJLT_t, DistMatrix_VC_STAR, SharedMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_STAR_VR, SHARED_MATRIX,
        sketch::FJLT_t, DistMatrix_STAR_VR, SharedMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX_STAR_VC, SHARED_MATRIX,
        sketch::FJLT_t, DistMatrix_STAR_VC, SharedMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FJLT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::FJLT_t, DistMatrix, DistMatrix,
        sketch::FJLT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        MATRIX, MATRIX,
        sketch::FastGaussianRFT_t, Matrix, Matrix,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::FastGaussianRFT_t, SparseMatrix, Matrix,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        SHARED_MATRIX, SHARED_MATRIX,
        sketch::FastGaussianRFT_t, SharedMatrix, SharedMatrix,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        ROOT_MATRIX, ROOT_MATRIX,
        sketch::FastGaussianRFT_t, RootMatrix, RootMatrix,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::FastGaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::FastGaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::FastGaussianRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastGaussianRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::FastGaussianRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::FastGaussianRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        MATRIX, MATRIX,
        sketch::FastMaternRFT_t, Matrix, Matrix,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        SPARSE_MATRIX, MATRIX,
        sketch::FastMaternRFT_t, SparseMatrix, Matrix,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        ROOT_MATRIX, ROOT_MATRIX,
        sketch::FastMaternRFT_t, RootMatrix, RootMatrix,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::FastMaternRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::FastMaternRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::FastMaternRFT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::FastMaternRFT_data_t);

    AUTO_APPLY_DISPATCH(FastMaternRFT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::FastMaternRFT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::FastMaternRFT_data_t);

#endif

#if SKYLARK_HAVE_FFTW

    AUTO_APPLY_DISPATCH(PPT,
        MATRIX, MATRIX,
        sketch::PPT_t, Matrix, Matrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        SPARSE_MATRIX, MATRIX,
        sketch::PPT_t, SparseMatrix, Matrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        SHARED_MATRIX, SHARED_MATRIX,
        sketch::PPT_t, SharedMatrix, SharedMatrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        ROOT_MATRIX, ROOT_MATRIX,
        sketch::PPT_t, RootMatrix, RootMatrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        DIST_MATRIX, DIST_MATRIX,
        sketch::PPT_t, DistMatrix, DistMatrix,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        DIST_MATRIX_VC_STAR, DIST_MATRIX_VC_STAR,
        sketch::PPT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        DIST_MATRIX_VR_STAR, DIST_MATRIX_VR_STAR,
        sketch::PPT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        DIST_MATRIX_STAR_VC, DIST_MATRIX_STAR_VC,
        sketch::PPT_t, DistMatrix_STAR_VC, DistMatrix_STAR_VC,
        sketch::PPT_data_t);

    AUTO_APPLY_DISPATCH(PPT,
        DIST_MATRIX_STAR_VR, DIST_MATRIX_STAR_VR,
        sketch::PPT_t, DistMatrix_STAR_VR, DistMatrix_STAR_VR,
        sketch::PPT_data_t);

#endif

#ifdef SKYLARK_HAVE_COMBBLAS

    // adding a bunch of sp -> sp_sketch -> dense types
    //FIXME: only tested types, */SOMETHING should work as well...
    AUTO_APPLY_DISPATCH(CWT,
        DIST_SPARSE_MATRIX, MATRIX,
        sketch::CWT_t, DistSparseMatrix, Matrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_SPARSE_MATRIX, DIST_MATRIX,
        sketch::CWT_t, DistSparseMatrix, DistMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_SPARSE_MATRIX, DIST_MATRIX_VC_STAR,
        sketch::CWT_t, DistSparseMatrix, DistMatrix_VC_STAR,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_SPARSE_MATRIX, DIST_MATRIX_VR_STAR,
        sketch::CWT_t, DistSparseMatrix, DistMatrix_VR_STAR,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_SPARSE_MATRIX, MATRIX,
        sketch::MMT_t, DistSparseMatrix, Matrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_SPARSE_MATRIX, DIST_MATRIX,
        sketch::MMT_t, DistSparseMatrix, DistMatrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_SPARSE_MATRIX, DIST_MATRIX_VC_STAR,
        sketch::MMT_t, DistSparseMatrix, DistMatrix_VC_STAR,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_SPARSE_MATRIX, DIST_MATRIX_VR_STAR,
        sketch::MMT_t, DistSparseMatrix, DistMatrix_VR_STAR,
        sketch::MMT_data_t);
#endif

#ifdef SKYLARK_HAVE_COMBBLAS

    AUTO_APPLY_DISPATCH(CWT,
        DIST_SPARSE_MATRIX, DIST_SPARSE_MATRIX,
        sketch::CWT_t, DistSparseMatrix, DistSparseMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(CWT,
        DIST_SPARSE_MATRIX, SPARSE_MATRIX,
        sketch::CWT_t, DistSparseMatrix, SparseMatrix,
        sketch::CWT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_SPARSE_MATRIX, DIST_SPARSE_MATRIX,
        sketch::MMT_t, DistSparseMatrix, DistSparseMatrix,
        sketch::MMT_data_t);

    AUTO_APPLY_DISPATCH(MMT,
        DIST_SPARSE_MATRIX, SPARSE_MATRIX,
        sketch::MMT_t, DistSparseMatrix, SparseMatrix,
        sketch::MMT_data_t);

#endif

    return 0;
}



} // extern "C"
