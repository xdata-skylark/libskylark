#ifndef SKYLARK_SKETCHC_HPP
#define SKYLARK_SKETCHC_HPP

#include "mpi.h"
#include "config.h"
#include "../../utility/distributions.hpp"

// TODO for now... but we can use the any,any to implement c-api
#define SKYLARK_NO_ANY
#include "../sketch.hpp"
#include "../../base/context.hpp"


// Some tools require special API declaration. Customizing the
// SKYLARK_EXTERN_API allows this. The default is simply nothing.
#ifndef SKYLARK_EXTERN_API
#define SKYLARK_EXTERN_API
#endif

namespace skylark {
namespace sketch {
namespace c {

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

enum matrix_type_t {
    MATRIX_TYPE_ERROR,
    MATRIX,                     /**< Dense Elemental matrix */
    SHARED_MATRIX,              /**< Same matrix on all processors: STAR-STAR */
    ROOT_MATRIX,                /**< One rank holds the matrix: CIRC-CIRC */
    DIST_MATRIX,                /**< Distributed Elemental matrix (MC-MR) */
    DIST_MATRIX_VC_STAR,
    DIST_MATRIX_VR_STAR,
    DIST_MATRIX_STAR_VC,
    DIST_MATRIX_STAR_VR,
    DIST_SPARSE_MATRIX,          /**< Sparse matrix (CombBLAS) */
    SPARSE_MATRIX                /**< Sparse local matrix */
};

struct sketch_transform_t {
    const transform_type_t type;
    sketch_transform_data_t * const transform_obj;

    sketch_transform_t(transform_type_t type,
        sketch_transform_data_t *transform_obj)
        : type(type), transform_obj(transform_obj) {}
};

} // namespace c
} // namespace sketch
} // namespace skylark

namespace base    = skylark::base;
namespace sketch  = skylark::sketch;
namespace sketchc = skylark::sketch::c;

#define SL_COLUMNWISE 1
#define SL_ROWWISE    2

extern "C" {

/** Returns a string describing the sketch transforms that are supported.
 */
SKYLARK_EXTERN_API char *sl_supported_sketch_transforms();

/** Converting an error code to a human readable string describing the failure
 *  @param errorcode the error code to resolve
 *  @return a string containing the error message
 */
SKYLARK_EXTERN_API const char* sl_strerror(const int errorcode);

/** Provide mechanism to check for add-ons.
 *  @return a boolean if the add-on library is enabled
 */
SKYLARK_EXTERN_API bool sl_has_elemental();
SKYLARK_EXTERN_API bool sl_has_combblas();


// Support for skylark::base::context_t.

/** Creating a default Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API int sl_create_default_context(int seed,
                                                 base::context_t **ctxt);

/** Creating a Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @param cm MPI communicator used
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API int sl_create_context(int seed, MPI_Comm cm,
                                         base::context_t **ctxt);

/** Free resources hold by a Skylark context.
 *  @param ctxt Skylark context
 */
SKYLARK_EXTERN_API int sl_free_context(base::context_t *ctxt);

/** Creating a sketch transformation.
 *  @param ctxt Sklark context
 *  @param type type of the sketch
 *  @param n input size
 *  @param s output size
 *  @return sketch transformation
 */
SKYLARK_EXTERN_API int sl_create_sketch_transform(
        base::context_t *ctxt, char *type,
        int n, int s, sketchc::sketch_transform_t **sketch, ...);

/** Deserialize a sketch transformation.
 *  @param data string of serialized JSON structure
 *  @param sketch the deserialized sketch transformation
 */
SKYLARK_EXTERN_API int sl_deserialize_sketch_transform(
       const char *data, sketchc::sketch_transform_t **sketch);

/** Serializes a sketch transformation.
 *  @param sketch the sketch to be serialized
 *  @param data of the serialized JSON structure
 */
SKYLARK_EXTERN_API int sl_serialize_sketch_transform(
       const sketchc::sketch_transform_t *sketch, char **data);

/** Free resources hold by a sketch transformation.
 *  @param S sketch transform
 */
SKYLARK_EXTERN_API int sl_free_sketch_transform(
        sketchc::sketch_transform_t *S);

/** Apply the sketch transformation to a matrix.
 *  @param S sketch transform
 *  @param input_type input matrix type
 *  @param A input matrix
 *  @param output_type output matrix type
 *  @param SA sketched matrix
 *  @param dim dimension on which to sketch (SL_COLUMNWISE/ROWWISE)
 */
SKYLARK_EXTERN_API int sl_apply_sketch_transform(
        sketchc::sketch_transform_t *S,
        char *input_type, void *A,
        char *output_type, void *SA, int dim);

// Helper functions to allow wrapping of object

SKYLARK_EXTERN_API int sl_wrap_raw_matrix(double *data, int m, int n, void **A);

SKYLARK_EXTERN_API int sl_free_raw_matrix_wrap(void *A_);

SKYLARK_EXTERN_API int sl_wrap_raw_sp_matrix(int *indptr, int *ind,
    double *data, int nnz, int n_rows, int n_cols, void **A);

SKYLARK_EXTERN_API int sl_free_raw_sp_matrix_wrap(void *A_);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_struct_updated(void *A_,
    bool *struct_updated);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_reset_update_flag(void *A_);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_nnz(void *A_, int *nnz);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_data(void *A_, int32_t *indptr,
        int32_t *indices, double *values);

} // extern "C"

#endif // SKYLARK_SKETCHC_HPP
