#ifndef SKYLARK_FJLT_HPP
#define SKYLARK_FJLT_HPP

#include "../config.h"
#include "FJLT_data.hpp"

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct FJLT_t {
    // To be specilized and derived.

};

} } /** namespace skylark::sketch */

#if SKYLARK_HAVE_ELEMENTAL && SKYLARK_HAVE_FFTW
# include "FJLT_Elemental.hpp"
#endif

#endif // SKYLARK_FJLT_HPP
