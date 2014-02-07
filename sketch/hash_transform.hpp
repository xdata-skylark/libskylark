#ifndef SKYLARK_HASH_TRANSFORM_HPP
#define SKYLARK_HASH_TRANSFORM_HPP

#include "../config.h"
#include "../utility/distributions.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class IdxDistributionType,
           template <typename> class ValueDistribution
        >
struct hash_transform_t { };

} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
#include "hash_transform_Elemental.hpp"
#endif

#if SKYLARK_HAVE_COMBBLAS
#include "hash_transform_CombBLAS.hpp"
#endif

#include "hash_transform_local_sparse.hpp"

#endif // SKYLARK_HASH_TRANSFORM_HPP
