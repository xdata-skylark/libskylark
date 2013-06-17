#ifndef EXACT_REGRESSOR_HPP
#define EXACT_REGRESSOR_HPP

namespace skylark {
namespace algorithms {

/* Tags for type of algorithms for exact regressor */
struct l2_solver_tag {};

struct qr_l2_solver_tag : l2_solver_tag {};

struct ne_l2_solver_tag : l2_solver_tag {};

struct svd_l2_solver_tag : l2_solver_tag {};

template <typename KrylovMethod>
struct iterative_l2_solver_tag : l2_solver_tag {};

/**
 * Regressor that solves the problem exactly (*as much as possible on a machine)
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressor. The top class is empty: real logic is in
 * specializations.
 */
template <typename RegressionType,
          typename MatrixType,
          typename RhsType,
          typename AlgTag = qr_l2_solver_tag>
class exact_regressor_t {
};


} // namespace sketch
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL
#include "exact_regressor_Elemental.hpp"
#endif

#endif // EXACT_REGRESSOR_HPP
