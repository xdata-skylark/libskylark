#ifndef SKYLARK_KRYLOV_ITER_PARAMS_HPP
#define SKYLARK_KRYLOV_ITER_PARAMS_HPP

#include <ostream>

namespace skylark { namespace algorithms {

struct krylov_iter_params_t {

    double tolerance;
    int iter_lim;
    bool am_i_printing;
    int log_level;
    int res_print;
    std::ostream& log_stream;
    int debug_level;

    krylov_iter_params_t(double tolerance = 1e-14,
        int iter_lim = 100,
        bool am_i_printing = 0,
        int log_level = 0,
        int res_print = 1,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) : tolerance(tolerance),
                               iter_lim(iter_lim),
                               am_i_printing(am_i_printing),
                               log_level(log_level),
                               res_print(res_print),
                               log_stream(log_stream),
                               debug_level(debug_level) {

  }

};

} } // namespace skylark::algorithms


#endif // SKYLARK_KRYLOV_ITER_PARAMS_HPP
