#ifndef SKYLARK_SKETCHED_REGRESSION_SOLVER_HPP
#define SKYLARK_SKETCHED_REGRESSION_SOLVER_HPP

#include "config.h"

namespace skylark {
namespace algorithms {

/**
 * Generic class for the sketch-and-solve strategy.
 */
template <typename RegressionProblemType,
          typename RhsType,
          typename SolType,
          typename SketchedRegressionType,
          typename SketchType,
          typename SketchRhsType,
          template <typename, typename> class SketchTransformType,
          typename ExactAlgTag>
class sketched_regression_solver_t {

};


} // namespace sketch
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL
#include "sketched_regression_solver_Elemental.hpp"
#endif

#endif // SKYLARK_SKETCHED_REGRESSION_SOLVER_HPP
