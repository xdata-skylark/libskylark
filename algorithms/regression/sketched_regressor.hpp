#ifndef SKETCHED_REGRESSOR_HPP
#define SKETCHED_REGRESSOR_HPP

namespace skylark {
namespace algorithms {

template <typename RegressionType,
          typename MatrixType,
          typename RhsType,
          typename SketchType,
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

#endif // SKETCHED_REGRESSOR_HPP
