#ifndef SKYLARK_EXACT_REGRESSOR_HPP
#define SKYLARK_EXACT_REGRESSOR_HPP

#include "../../config.h"

namespace skylark {
namespace algorithms {

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
 * @tparam RegressionProblemType Type of regression problem solved.
 * @tparam RhsType Right-hand side matrix type.
 * @tparam SolType Solution matrix type.
 * @tparam AlgTag Tag specifying the algorithm used (tags differ based on problem).
 */
template <typename RegressionProblemType,
          typename RhsType,
          typename SolType,
          typename AlgTag>
class exact_regressor_t {

};


} // namespace algorithms
} // namespace skylark


#include "linearl2_exact_regressor.hpp"

#endif // SKYLARK_EXACT_REGRESSOR_HPP
