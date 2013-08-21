#ifndef EXACT_REGRESSOR_HPP
#define EXACT_REGRESSOR_HPP

namespace skylark {
namespace algorithms {

/// Base class for tags specifying strategies for solving L2 regression problems.
struct l2_solver_tag {};

/// Tag for using QR to solve L2 regression problems.
struct qr_l2_solver_tag : l2_solver_tag {};

/// Tag for using normal equations to solve L2 regression problems.
struct ne_l2_solver_tag : l2_solver_tag {};

/// Tag for using SVD to solve L2 regression problems.
struct svd_l2_solver_tag : l2_solver_tag {};

/**
 * Tag for using an iterative method to solve L2 regression.
 *
 * @tparam KrylovMethod The underlying Krylov method used.
 */
template <typename KrylovMethod>
struct iterative_l2_solver_tag : l2_solver_tag {};

/// Tag for all krylov methods to inherit from
struct krylov_tag {};

/// Tag for using LSQR
struct lsqr_tag: public krylov_tag {};

/**
 * Regressor that solves the problem exactly (as much as possible on a machine).
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressor. The top class is empty: real logic is in
 * specializations.
 *
 * @tparam RegressionType Tag specificying regression type, e.g. l2_tag.
 * @tparam MatrixType Input matrix type.
 * @tparam RhsType Right-hand side matrix type.
 * @tparam AlgTag Tag specifyin the algorithm used.
 */
template <typename RegressionType,
          typename MatrixType,
          typename RhsType,
          typename AlgTag = qr_l2_solver_tag>
class exact_regressor_t {
};


} // namespace sketch
} // namespace skylark

#ifdef SKYLARK_HAVE_ELEMENTAL
#include "exact_regressor_Elemental.hpp"
#endif

#include "exact_regressor_Krylov.hpp"

#endif // EXACT_REGRESSOR_HPP
