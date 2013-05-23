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
    DIST_MATRIX_VR_STAR
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

/** Support for skylark::sketch::context_t. */
SKYLARK_EXTERN_API sketch::context_t *sl_create_default_context(int seed);
SKYLARK_EXTERN_API sketch::context_t *sl_create_context(int seed, MPI_Comm cm);
SKYLARK_EXTERN_API void sl_free_context(sketch::context_t *ctxt);
SKYLARK_EXTERN_API int sl_context_rank(sketch::context_t *ctxt);
SKYLARK_EXTERN_API int sl_context_size(sketch::context_t *ctxt);


/** Transforms */
SKYLARK_EXTERN_API sketchc::sketch_transform_t *sl_create_sketch_transform(
        sketch::context_t *ctxt, char *type, char *input, char *output,
        int n, int s, ...);
SKYLARK_EXTERN_API void sl_free_sketch_transform(
        sketchc::sketch_transform_t *S);
SKYLARK_EXTERN_API void sl_apply_sketch_transform(
        sketchc::sketch_transform_t *S, void *A, void *SA, int dim);

} // extern "C"

#endif // SKETCHC_HPP
