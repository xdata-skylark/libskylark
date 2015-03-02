#ifndef SKYLARK_KRYLOV_ITER_PARAMS_HPP
#define SKYLARK_KRYLOV_ITER_PARAMS_HPP

#include <ostream>

namespace skylark { namespace algorithms {

struct krylov_iter_params_t : public base::params_t {

    double tolerance;
    int iter_lim;
    int res_print;

    krylov_iter_params_t(double tolerance = 1e-14,
        int iter_lim = 100,
        bool am_i_printing = 0,
        int log_level = 0,
        int res_print = 1,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, debug_level),
        tolerance(tolerance),
        iter_lim(iter_lim),
        res_print(res_print) {

  }

};

} } // namespace skylark::algorithms


#endif // SKYLARK_KRYLOV_ITER_PARAMS_HPP
