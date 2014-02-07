#ifndef SKETCHC_HPP
#define SKETCHC_HPP

#include "mpi.h"
#include "../../config.h"
#include "../../utility/distributions.hpp"

#include "../sketch.hpp"

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
    GaussianRFT,
    LaplacianRFT,
    FastGaussianRFT
};

enum matrix_type_t {
    MATRIX_TYPE_ERROR,
    MATRIX,                     /**< Dense Elemental matrix */
    DIST_MATRIX,                /**< Distributed Elemental matrix (MC-MR) */
    DIST_MATRIX_VC_STAR,
    DIST_MATRIX_VR_STAR,
    DIST_MATRIX_STAR_VC,
    DIST_MATRIX_STAR_VR,
    DIST_SPARSE_MATRIX          /**< Sparse matrix (CombBLAS) */
};

struct sketch_transform_t {
    const transform_type_t type;
    void * const transform_obj;

    sketch_transform_t(
        transform_type_t type,void *transform_obj)
        : type(type), transform_obj(transform_obj) {}
};

} // namespace c
} // namespace sketch
} // namespace skylark

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


// Support for skylark::sketch::context_t.

/** Creating a default Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API int sl_create_default_context(int seed,
                                                 sketch::context_t **ctxt);

/** Creating a Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @param cm MPI communicator used
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API int sl_create_context(int seed, MPI_Comm cm,
                                         sketch::context_t **ctxt);

/** Free resources hold by a Skylark context.
 *  @param ctxt Skylark context
 */
SKYLARK_EXTERN_API int sl_free_context(sketch::context_t *ctxt);

/** Get rank.
 *  @param ctxt Skylark context
 *  @return MPI rank
 */
SKYLARK_EXTERN_API int sl_context_rank(sketch::context_t *ctxt, int *rank);

/** Get total number of processor.
 *  @param ctxt Skylark context
 *  @return number of processors in the context
 */
SKYLARK_EXTERN_API int sl_context_size(sketch::context_t *ctxt, int *size);

/** Creating a sketch transformation.
 *  @param ctxt Sklark context
 *  @param type type of the sketch
 *  @param n input size
 *  @param s output size
 *  @return sketch transformation
 */
SKYLARK_EXTERN_API int sl_create_sketch_transform(
        sketch::context_t *ctxt, char *type, 
        int n, int s, sketchc::sketch_transform_t **sketch, ...);

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


} // extern "C"

#endif // SKETCHC_HPP
