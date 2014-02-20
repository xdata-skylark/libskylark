#ifndef SKYLARK_PPT_HPP
#define SKYLARK_PPT_HPP

#include "PPT_data.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct PPT_t {
    // To be specilized and derived.

};

} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL
# include "PPT_Elemental.hpp"
#endif

#endif // SKYLARK_PPT_HPP
