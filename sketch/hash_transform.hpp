#ifndef SKYLARK_HASH_TRANSFORM_HPP
#define SKYLARK_HASH_TRANSFORM_HPP

#include "../config.h"
#include "../utility/distributions.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType,
           typename IdxDistributionType,
           template <typename> class ValueDistributionType
        >
struct hash_transform_t { };

} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
#include "hash_transform_Elemental.hpp"
#endif

#if SKYLARK_HAVE_COMBBLAS
#include "hash_transform_CombBLAS.hpp"
#endif

#endif // SKYLARK_HASH_TRANSFORM_HPP
