#ifndef SKYLARK_REGRESSION_SOLVER_HPP
#define SKYLARK_REGRESSION_SOLVER_HPP

#include "config.h"

namespace skylark {
namespace algorithms {

/**
 * Classical regression solver that solves the original problem.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressor. The top class is empty: real logic is in
 * specializations.
 *
 * @tparam RegressionProblemType Type of regression problem solved.
 * @tparam RhsType Right-hand side matrix type.
 * @tparam SolType Solution matrix type.
 * @tparam AlgTag Tag specifying the algorithm used (tags differ based on problem).
 */
template <typename RegressionProblemType,
          typename RhsType,
          typename SolType,
          typename AlgTag>
class regression_solver_t {

};


} // namespace algorithms
} // namespace skylark


#include "linearl2_regression_solver.hpp"

#endif // SKYLARK_REGRESSION_SOLVER_HPP
