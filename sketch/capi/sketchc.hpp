#ifndef SKETCHC_HPP
#define SKETCHC_HPP

#include "mpi.h"
#include "skylark.hpp"

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
    LaplacianRFT
};

enum matrix_type_t {
    MATRIX_TYPE_ERROR,
    MATRIX,                     /**< Dense elemental matrix */
    DIST_MATRIX_VC_STAR,
    DIST_MATRIX_VR_STAR,
    DIST_SPARSE_MATRIX          /**< Sparse matrix (CombBLAS) */
};

struct sketch_transform_t {
    const transform_type_t type;
    const matrix_type_t input;
    const matrix_type_t output;
    void * const transform_obj;

    sketch_transform_t(
        transform_type_t type, matrix_type_t input,
        matrix_type_t output, void *transform_obj)
        : type(type), input(input),
          output(output), transform_obj(transform_obj) {}
};

} // namespace c
} // namespace sketch
} // namespace skylark

namespace sketch  = skylark::sketch;
namespace sketchc = skylark::sketch::c;

#define SL_COLUMNWISE 1
#define SL_ROWWISE    2

extern "C" {

// Support for skylark::sketch::context_t.

/** Creating a default Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API sketch::context_t *sl_create_default_context(int seed);

/** Creating a Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @param cm MPI communicator used
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API sketch::context_t *sl_create_context(int seed, MPI_Comm cm);

/** Free resources hold by a Skylark context.
 *  @param ctxt Skylark context
 */
SKYLARK_EXTERN_API void sl_free_context(sketch::context_t *ctxt);

/** Get rank.
 *  @param ctxt Skylark context
 *  @return MPI rank
 */
SKYLARK_EXTERN_API int sl_context_rank(sketch::context_t *ctxt);
/** Get total number of processor.
 *  @param ctxt Skylark context
 *  @return number of processors in the context
 */
SKYLARK_EXTERN_API int sl_context_size(sketch::context_t *ctxt);

// Transforms
/** Creating a sketch transformation.
 *  @param ctxt Sklark context
 *  @param type type of the sketch
 *  @param input input matrix type
 *  @param output output matrix type
 *  @param n input size
 *  @param s output size
 *  @return sketch transformation
 */
SKYLARK_EXTERN_API sketchc::sketch_transform_t *sl_create_sketch_transform(
        sketch::context_t *ctxt, char *type, char *input, char *output,
        int n, int s, ...);

/** Free resources hold by a sketch transformation.
 *  @param S sketch transform
 */
SKYLARK_EXTERN_API void sl_free_sketch_transform(
        sketchc::sketch_transform_t *S);

/** Apply the sketch transformation to a matrix.
 *  @param S sketch transform
 *  @param A input matrix
 *  @param AS sketched matrix
 *  @param dim direction of sketch application
 */
SKYLARK_EXTERN_API void sl_apply_sketch_transform(
        sketchc::sketch_transform_t *S, void *A, void *SA, int dim);

// Helper functions to allow wrapping of object

SKYLARK_EXTERN_API void *sl_wrap_raw_matrix(double *A, int m, int n);

SKYLARK_EXTERN_API void sl_free_raw_matrix_wrap(void *A_);


} // extern "C"

#endif // SKETCHC_HPP
