#ifndef SKETCHED_REGRESSOR_ELEMENTAL_HPP
#define SKETCHED_REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "config.h"
#include "regression_problem.hpp"
#include "sketch/sketch.hpp"


namespace skylark {
namespace algorithms {

/**
 * Sketched l2 regressor using sketch-and-solve.
 *
 * This implementation assumes RhsType is the same as MatrixType.
 */
template <typename MatrixType,
          typename SketchType,
          template <typename, typename> class TransformType,
          typename ExactAlgTag>
class sketched_regressor_t<l2_tag,
                           MatrixType,
                           MatrixType,
                           SketchType,
                           TransformType,
                           ExactAlgTag,
                           sketch_and_solve_tag> {

    typedef regression_problem_t<l2_tag, MatrixType> problem_type;
    typedef regression_problem_t<l2_tag, SketchType> sketched_problem_type;

    typedef exact_regressor_t<l2_tag,
                              SketchType,
                              SketchType,
                              ExactAlgTag> underlying_regressor_type;

    typedef TransformType<MatrixType, SketchType> sketch_transform_type;

private:
    int _sketch_size;
    sketch_transform_type *_sketch_transform;
    underlying_regressor_type  *_underlying_regressor;

public:
    sketched_regressor_t(const problem_type &problem, int sketch_size,
        sketch::context_t& context) : _sketch_size(sketch_size) {
        // TODO m < n
        _sketch_transform =
            new sketch_transform_type(problem.m, sketch_size, context);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        SketchType sketch(sketch_size, problem.n);
        _sketch_transform->apply(problem.input_matrix, sketch,
            sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_regressor = new underlying_regressor_type(sketched_problem);
    }

    ~sketched_regressor_t() {
        delete _underlying_regressor;
        delete _sketch_transform;
    }

    void solve(const MatrixType &b, SketchType &x) {
        // TODO For DistMatrix this will allocate on DefaultGrid...
        //      Want to make constructor independent.
        SketchType Sb(_sketch_size, 1);
        _sketch_transform->apply(b, Sb, sketch::columnwise_tag());
        _underlying_regressor->solve(Sb, x);
    }

    void solve_mulitple(const MatrixType &B, SketchType &X) {
        // TODO For DistMatrix this will allocate on DefaultGrid... 
        //      Want to make constructor independent.
        SketchType SB(_sketch_size, B.Width());
        _sketch_transform->apply(SB, SB, sketch::columnwise_tag());
        _underlying_regressor->solve_mulitple(SB, X);
    }
};

} // namespace algorithms
} // namespace skylark

#endif // SKETCHED_REGRESSOR_ELEMENTAL_HPP
