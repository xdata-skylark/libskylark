#ifndef SKYLARK_FAST_EXACT_REGRESSOR_HPP
#define SKYLARK_FAST_EXACT_REGRESSOR_HPP

#include "../../config.h"

namespace skylark {
namespace algorithms {

/**
 * Regressors on the original problem that have been accelerated using
 * sketching.
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
 * @tparam AlgTag Tag specifying the algorithm used (tags differ based on
 *                problem).
 */
template <typename RegressionProblemType,
          typename RhsType,
          typename SolType,
          typename AlgTag>
class fast_exact_regressor_t {

};


} // namespace algorithms
} // namespace skylark


#include "fast_linearl2_exact_regressor.hpp"

#endif // SKYLARK_FAST_EXACT_REGRESSOR_HPP
