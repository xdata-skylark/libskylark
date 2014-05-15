#ifndef SKYLARK_LINEARL2_REGRESSION_SOLVER_HPP
#define SKYLARK_LINEARL2_REGRESSION_SOLVER_HPP

#include "regression_solver.hpp"

namespace skylark {
namespace algorithms {

/// Base class for tags specifying strategies for solving L2
/// linear regression problems.
struct l2_solver_tag {};

/// Tag for using QR to solve L2 linear regression problems.
struct qr_l2_solver_tag : l2_solver_tag {};

/// Tag for using semi-normal equations to solve L2 linear regression problems.
struct sne_l2_solver_tag : l2_solver_tag {};

/// Tag for using normal equations to solve L2 linear regression problems.
struct ne_l2_solver_tag : l2_solver_tag {};

/// Tag for using SVD to solve L2 linear regression problems.
struct svd_l2_solver_tag : l2_solver_tag {};

/**
 * Tag for using an iterative method to solve L2 linear regression.
 *
 * @tparam KrylovMethod The underlying Krylov method used.
 */
template <typename KrylovMethod>
struct iterative_l2_solver_tag : l2_solver_tag {};

/// Tag for all krylov methods to inherit from
struct krylov_tag {};

/// Tag for using LSQR
struct lsqr_tag: public krylov_tag {};

} // namespace algorithms
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL
#include "linearl2_regression_solver_Elemental.hpp"
#endif

#include "linearl2_regression_solver_Krylov.hpp"

#endif // SKYLARK_LINEARL2_REGRESSION_SOLVER_HPP
