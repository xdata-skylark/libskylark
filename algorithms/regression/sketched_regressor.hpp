#ifndef SKYLARK_SKETCHED_REGRESSOR_HPP
#define SKYLARK_SKETCHED_REGRESSOR_HPP

#include "../../config.h"

namespace skylark {
namespace algorithms {

template <typename RegressionProblemType,
          typename RhsType,
          typename SolType,
          typename SketchedRegressionType,
          typename SketchType,
          typename SketchRhsType,
          template <typename, typename> class SketchTransformType,
          typename ExactAlgTag,
          typename UseTag = sketch_and_solve_tag>
class sketched_regressor_t {

};


} // namespace sketch
} // namespace skylark

#if SKYLARK_HAVE_ELEMENTAL
#include "sketched_regressor_Elemental.hpp"
#endif

#endif // SKYLARK_SKETCHED_REGRESSOR_HPP
