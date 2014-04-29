#ifndef SKYLARK_ITER_PARAMS_HPP
#define SKYLARK_ITER_PARAMS_HPP

namespace skylark { namespace nla {

struct iter_params_t {

  double tolerance;
  bool am_i_printing;
  int iter_lim;
  int debug_level;
  int return_code;

  iter_params_t(double tolerance = 1e-14,
                bool am_i_printing = 0,
                int iter_lim = 100,
                int debug_level = 0) : tolerance(tolerance),
                                   am_i_printing(am_i_printing),
                                   iter_lim(iter_lim),
                                   debug_level(debug_level),
                                   return_code(0) {}

};

} } // namespace skylark::nla


#endif // SKYLARK_ITER_PARAMS_HPP
