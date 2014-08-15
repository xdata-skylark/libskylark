#ifndef SKYLARK_RFUT_HPP
#define SKYLARK_RFUT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

/**
 * Random Fast Unitary Transform.

 * This is a "transform", not a "sketching transform", so it's output
 * is always the same format as its input.
 */
template <typename MatrixType, typename FUT, typename ValueDistributionType>
struct RFUT_t {

};

} } /** namespace skylark::sketch */

# include "RFUT_Elemental.hpp"

#endif // RFUT
