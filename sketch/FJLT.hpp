#ifndef FJLT_HPP
#define FJLT_HPP

#include "../config.h"

namespace skylark {
namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct FJLT_t {

};

} // namespace sketch
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL && SKYLARK_HAVE_FFTW
# include "FJLT_Elemental.hpp"
#endif

#endif // FJLT_HPP
