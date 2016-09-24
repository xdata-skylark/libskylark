#ifndef SKYLARK_IO_HPP
#define SKYLARK_IO_HPP

namespace skylark { namespace utility { namespace io {
enum fileformat_t : int {
    FORMAT_LIBSVM = 0,
    FORMAT_HDF5 = 1
};

} } }

#include "libsvm_io.hpp"
#include "arc_list.hpp"

#ifdef SKYLARK_HAVE_HDF5
#include "hdf5_io.hpp"
#endif

#endif
