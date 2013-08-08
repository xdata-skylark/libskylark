#ifndef SPARSET_HPP
#define SPARSET_HPP

#include "config.h"

#include "utility/distributions.hpp"

namespace skylark {
namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType,
           typename IdxDistributionType,
           template <typename> class ValueDistributionType
        >
struct hash_transform_t {
};

} // namespace sketch
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL
    #include "sparset_Elemental.hpp"
#endif

#ifdef SKYLARK_HAVE_COMBBLAS
    #include "sparset_CombBLAS.hpp"
#endif

#endif // SPARSET_HPP
