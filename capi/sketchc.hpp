#ifndef SKYLARK_SKETCHC_HPP
#define SKYLARK_SKETCHC_HPP

#include "mpi.h"
#include "config.h"

#include "basec.hpp"

namespace skylark {
namespace sketch {

class sketch_transform_data_t;

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

struct sketch_transform_t {
    const transform_type_t type;
    sketch_transform_data_t * const transform_obj;

    sketch_transform_t(transform_type_t type,
        sketch_transform_data_t *transform_obj)
        : type(type), transform_obj(transform_obj) {}
};

} // namespace c
} // namespace sketchc

} // namespace skylark


namespace base    = skylark::base;
namespace sketch  = skylark::sketch;
namespace sketchc = skylark::sketch::c;

extern "C" {

/** Creating a sketch transformation.
 *  @param ctxt Sklark context
 *  @param type type of the sketch
 *  @param n input size
 *  @param s output size
 *  @param sketch
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


} // extern "C"

#endif // SKYLARK_SKETCHC_HPP
