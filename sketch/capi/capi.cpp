#include "skylark.hpp"
#include "sketch/capi/sketchc.hpp"

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
    STRCMP_TYPE(DistMatrix_VC_STAR, sketchc::DIST_MATRIX_VC_STAR);
    STRCMP_TYPE(DistMatrix_VR_STAR, sketchc::DIST_MATRIX_VR_STAR);
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
    STRCMP_TYPE(GaussianRFT, sketchc::GaussianRFT);
    STRCMP_TYPE(LaplacianRFT, sketchc::LaplacianRFT);

    return sketchc::TRANSFORM_TYPE_ERROR;
}

// Just for shorter notation
#if SKYLARK_HAVE_ELEMENTAL
typedef elem::Matrix<double> Matrix;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> DistMatrix_VR_STAR;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistMatrix_VC_STAR;
#endif
#ifdef SKYLARK_HAVE_COMBBLAS
typedef SpDCCols< size_t, double > col_t;
typedef SpParMat< size_t, double, col_t > DistSparseMatrix_t;
#endif


extern "C" {

/** Support for skylark::sketch::context_t. */
SKYLARK_EXTERN_API sketch::context_t *sl_create_default_context(int seed) {
  return new sketch::context_t(seed, boost::mpi::communicator());
}

SKYLARK_EXTERN_API sketch::context_t *sl_create_context(int seed,
  MPI_Comm comm) {
  return new sketch::context_t(seed,
    boost::mpi::communicator(comm, boost::mpi::comm_duplicate));
}

SKYLARK_EXTERN_API void sl_free_context(sketch::context_t *ctxt) {
  delete ctxt;
}

SKYLARK_EXTERN_API int sl_context_rank(sketch::context_t *ctxt) {
  return ctxt->rank;
}

SKYLARK_EXTERN_API int sl_context_size(sketch::context_t *ctxt) {
  return ctxt->size;
}

/** Transforms */
SKYLARK_EXTERN_API sketchc::sketch_transform_t
   *sl_create_sketch_transform(sketch::context_t *ctxt,
     char *type_, char *input_, char *output_, int n, int s, ...) {

  sketchc::transform_type_t type = str2transform_type(type_);
  sketchc::matrix_type_t input   = str2matrix_type(input_);
  sketchc::matrix_type_t output  = str2matrix_type(output_);

# define AUTO_NEW_DISPATCH(T, I, O, C, IT, OT)                      \
  if (type == T && input == I && output == O)                       \
    return new sketchc::sketch_transform_t(                         \
                                           type, input, output,     \
                                             new C<IT, OT>(n, s, *ctxt));

# define AUTO_NEW_DISPATCH_1P(T, I, O, C, IT, OT)                   \
  if (type == T && input == I && output == O)  {                    \
      va_list argp;                                                 \
      va_start(argp, s);                                            \
      double p1 = va_arg(argp, double);                             \
      sketchc::sketch_transform_t *r =                              \
          new sketchc::sketch_transform_t(type, input, output,      \
              new C<IT, OT>(n, s, p1, *ctxt));                      \
      va_end(argp);                                                 \
      return r;                                                     \
  }

#if SKYLARK_HAVE_ELEMENTAL

  AUTO_NEW_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::JLT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_NEW_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::JLT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_NEW_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_NEW_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_NEW_DISPATCH_1P(sketchc::CT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::CT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_NEW_DISPATCH_1P(sketchc::CT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::CT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_NEW_DISPATCH_1P(sketchc::CT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::CT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_NEW_DISPATCH_1P(sketchc::CT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::CT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_NEW_DISPATCH(sketchc::CWT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::CWT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_NEW_DISPATCH(sketchc::CWT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::CWT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_NEW_DISPATCH(sketchc::MMT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::MMT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_NEW_DISPATCH(sketchc::MMT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::MMT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_NEW_DISPATCH_1P(sketchc::WZT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::WZT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_NEW_DISPATCH_1P(sketchc::WZT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::WZT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_NEW_DISPATCH_1P(sketchc::GaussianRFT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_NEW_DISPATCH_1P(sketchc::GaussianRFT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_NEW_DISPATCH_1P(sketchc::LaplacianRFT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_NEW_DISPATCH_1P(sketchc::LaplacianRFT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

#if SKYLARK_HAVE_FFTW

  AUTO_NEW_DISPATCH(sketchc::FJLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::FJLT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_NEW_DISPATCH(sketchc::FJLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::FJLT_t, DistMatrix_VC_STAR, Matrix);

#endif
#ifdef SKYLARK_HAVE_COMBBLAS

  AUTO_NEW_DISPATCH(sketchc::CWT,
      sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
      sketch::CWT_t, DistSparseMatrix_t, DistSparseMatrix_t);

#endif
#endif

  // TODO error handling
  return NULL;

}

SKYLARK_EXTERN_API
    void sl_free_sketch_transform(sketchc::sketch_transform_t *S) {

  sketchc::transform_type_t type = S->type;
  sketchc::matrix_type_t input   = S->input;
  sketchc::matrix_type_t output  = S->output;

# define AUTO_DELETE_DISPATCH(T, I, O, C, IT, OT)   \
  if (type == T && input == I && output == O)       \
    delete static_cast<C<IT, OT> *>(S->transform_obj);

#if SKYLARK_HAVE_ELEMENTAL

  AUTO_DELETE_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::JLT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::JLT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_DELETE_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_DELETE_DISPATCH(sketchc::CT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::CT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::CT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::CT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::CT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::CT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_DELETE_DISPATCH(sketchc::CT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::CT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_DELETE_DISPATCH(sketchc::CWT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::CWT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::CWT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::CWT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::MMT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::MMT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::MMT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::MMT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::WZT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::WZT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::WZT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::WZT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::GaussianRFT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_DELETE_DISPATCH(sketchc::GaussianRFT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_DELETE_DISPATCH(sketchc::LaplacianRFT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_DELETE_DISPATCH(sketchc::LaplacianRFT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

#if SKYLARK_HAVE_FFTW

  AUTO_DELETE_DISPATCH(sketchc::FJLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::FJLT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_DELETE_DISPATCH(sketchc::FJLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::FJLT_t, DistMatrix_VC_STAR, Matrix);

#endif
#ifdef SKYLARK_HAVE_COMBBLAS

  AUTO_DELETE_DISPATCH(sketchc::CWT,
      sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
      sketch::CWT_t, DistSparseMatrix_t, DistSparseMatrix_t);

#endif
#endif

  // Now can delete object
  delete S;
}

SKYLARK_EXTERN_API void
   sl_apply_sketch_transform(sketchc::sketch_transform_t *S_,
     void *A_, void *SA_, int dim) {

  sketchc::transform_type_t type = S_->type;
  sketchc::matrix_type_t input   = S_->input;
  sketchc::matrix_type_t output  = S_->output;

# define AUTO_APPLY_DISPATCH(T, I, O, C, IT, OT)                        \
  if (type == T && input == I && output == O) {                         \
    C<IT, OT> &S = * static_cast<C<IT, OT>*>(S_->transform_obj);        \
    IT &A = * static_cast<IT*>(A_);                                     \
    OT &SA = * static_cast<OT*>(SA_);                                   \
                                                                        \
    if (dim == 1)                                                       \
      S.apply(A, SA, sketch::columnwise_tag());                         \
    if (dim == 2)                                                       \
      S.apply(A, SA, sketch::rowwise_tag());                           \
  }

#if SKYLARK_HAVE_ELEMENTAL

  AUTO_APPLY_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::JLT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::JLT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::JLT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_APPLY_DISPATCH(sketchc::JLT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::JLT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_APPLY_DISPATCH(sketchc::CT,
      sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
      sketch::CT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::CT,
      sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
      sketch::CT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::CT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::CT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_APPLY_DISPATCH(sketchc::CT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::CT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_APPLY_DISPATCH(sketchc::CWT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::CWT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::CWT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::CWT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::MMT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::MMT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::MMT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::MMT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::WZT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
    sketch::WZT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::WZT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
    sketch::WZT_t, DistMatrix_VC_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::GaussianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_APPLY_DISPATCH(sketchc::GaussianRFT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::GaussianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

  AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
    sketchc::DIST_MATRIX_VR_STAR, sketchc::DIST_MATRIX_VR_STAR,
    sketch::LaplacianRFT_t, DistMatrix_VR_STAR, DistMatrix_VR_STAR);

  AUTO_APPLY_DISPATCH(sketchc::LaplacianRFT,
    sketchc::DIST_MATRIX_VC_STAR, sketchc::DIST_MATRIX_VC_STAR,
    sketch::LaplacianRFT_t, DistMatrix_VC_STAR, DistMatrix_VC_STAR);

#if SKYLARK_HAVE_FFTW

  AUTO_APPLY_DISPATCH(sketchc::FJLT,
      sketchc::DIST_MATRIX_VR_STAR, sketchc::MATRIX,
      sketch::FJLT_t, DistMatrix_VR_STAR, Matrix);

  AUTO_APPLY_DISPATCH(sketchc::FJLT,
      sketchc::DIST_MATRIX_VC_STAR, sketchc::MATRIX,
      sketch::FJLT_t, DistMatrix_VC_STAR, Matrix);

#endif
#ifdef SKYLARK_HAVE_COMBBLAS

  AUTO_APPLY_DISPATCH(sketchc::CWT,
      sketchc::DIST_SPARSE_MATRIX, sketchc::DIST_SPARSE_MATRIX,
      sketch::CWT_t, DistSparseMatrix_t, DistSparseMatrix_t);

#endif
#endif

}

SKYLARK_EXTERN_API void *sl_wrap_raw_matrix(double *data, int m, int n)  {
    Matrix *A = new Matrix();
    A->Attach(m, n, data, m);
    return A;
}

SKYLARK_EXTERN_API void sl_free_raw_matrix_wrap(void *A_) {
    delete static_cast<Matrix *>(A_);
}

} // extern "C"
