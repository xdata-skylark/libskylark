#ifndef SKYLARK_TRANSFORMS_HPP
#define SKYLARK_TRANSFORMS_HPP

namespace skylark {
namespace sketch {

/** Define a base class for tagging which dimension you are sketching/transforming */
struct dimension_tag {};

/// Apply the sketch/transform to the columns. In matrix form this is A->SA.
struct columnwise_tag : dimension_tag {};

/// Apply the sketch/transform to the rows. In matrix form this is A->AS^T
struct rowwise_tag : dimension_tag {};

/**
 * TWO SIDED: This can be thought of as a composition operation. Therefore,
 * there is no need to separately include this in config!
 */


} } /** namespace skylark::sketch */

#endif // TRANSFORMS_HPP
