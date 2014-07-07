#ifndef SKYLARK_ITER_PARAMS_HPP
#define SKYLARK_ITER_PARAMS_HPP

#include <ostream>

namespace skylark { namespace algorithms {

struct iter_params_t {

    double tolerance;
    bool am_i_printing;
    int iter_lim;
    int log_level;
    std::ostream& log_stream;
    int debug_level;

    iter_params_t(double tolerance = 1e-14,
        bool am_i_printing = 0,
        int iter_lim = 100,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) : tolerance(tolerance),
                               am_i_printing(am_i_printing),
                               iter_lim(iter_lim),
                               log_level(log_level),
                               log_stream(log_stream),
                               debug_level(debug_level) {

  }

};

} } // namespace skylark::nla


#endif // SKYLARK_ITER_PARAMS_HPP
