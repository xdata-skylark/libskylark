#ifndef SKYLARK_ASY_ITER_PARAMS_HPP
#define SKYLARK_ASY_ITER_PARAMS_HPP

#include <ostream>

namespace skylark { namespace algorithms {

struct asy_iter_params_t : public base::params_t {

    double tolerance;    /**< Tolerance on ||Ax-||_2/||b||_2 for convergence.
                              If <=0 then no convergence test is done. */
    int syn_sweeps;      /**< How many sweeps to do before a synchronization.
                              A value of <= 0 will just do sweeps_lim, and not
                              test synchronize at all.
                              If tolerance > 0 then convergence is tested in
                              every synchronization. */
    int sweeps_lim;      /**< Max amount of sweeps for a pure asychronous
                              method; number of internal preconditioner sweeps
                              in  flexible method. */

    // Parameters for an outer flexible Krylov method
    int iter_lim;
    int iter_res_print;

    asy_iter_params_t(double tolerance = 1e-3,
        int syn_sweeps = 10,
        int sweeps_lim = 100,
        int iter_lim = 20,
        int iter_res_print = 1,
        bool am_i_printing = 0,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, debug_level),
        tolerance(tolerance),
        syn_sweeps(syn_sweeps),
        sweeps_lim(sweeps_lim),
        iter_lim(iter_lim),
        iter_res_print(iter_res_print) {
    }

};

} } // namespace skylark::algorithms


#endif // SKYLARK_ASY_ITER_PARAMS_HPP
