#ifndef SKYLARK_TRANSFORMS_HPP
#define SKYLARK_TRANSFORMS_HPP

#include "sketch_transform_data.hpp"

namespace skylark {
namespace sketch {

/** Define a base class for tagging which dimension you are sketching/transforming */
struct dimension_tag {};

/// Apply the sketch/transform to the columns. In matrix form this is A->SA.
struct columnwise_tag : dimension_tag {};

/// Apply the sketch/transform to the rows. In matrix form this is A->AS^T
struct rowwise_tag : dimension_tag {};

} } /** namespace skylark::sketch */

#endif // TRANSFORMS_HPP
