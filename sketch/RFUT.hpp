#ifndef RFUT_HPP
#define RFUT_HPP

#include "../config.h"

namespace skylark {
namespace sketch {

/**
 * Random Fast Unitary Transform.

 * This is a "transform", not a "sketching transform", so it's output
 * is always the same format as its input.
 */
template <typename MatrixType, typename FUT, typename ValueDistributionType>
struct RFUT_t {

};

} // namespace sketch
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL
# include "RFUT_Elemental.hpp"
#endif

#endif // RFUT
