#ifndef SKYLARK_PARAMS_HPP
#define SKYLARK_PARAMS_HPP

#include <ostream>

namespace skylark { namespace base {

/**
 * Base parameter structure for parameter structures used througout Skylark.
 */
struct params_t {

    bool am_i_printing;
    int log_level;
    std::ostream& log_stream;
    int debug_level;

    params_t(bool am_i_printing = false,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) : am_i_printing(am_i_printing),
                               log_level(log_level),
                               log_stream(log_stream),
                               debug_level(debug_level) {

  }

};

} } // namespace skylark::base


#endif // SKYLARK_PARAMS_HPP
