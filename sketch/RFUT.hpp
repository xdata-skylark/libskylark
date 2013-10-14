#ifndef SKYLARK_RFUT_HPP
#define SKYLARK_RFUT_HPP

#include "../config.h"

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

#if SKYLARK_HAVE_ELEMENTAL
# include "RFUT_Elemental.hpp"
#endif

#endif // RFUT
