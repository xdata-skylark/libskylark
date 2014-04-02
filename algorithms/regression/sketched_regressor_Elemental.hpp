#ifndef SKETCHED_REGRESSOR_ELEMENTAL_HPP
#define SKETCHED_REGRESSOR_ELEMENTAL_HPP

#include <boost/mpi.hpp>
#include <elemental.hpp>

#include "regression_problem.hpp"
#include "../../base/context.hpp"


namespace skylark {
namespace algorithms {

/**
 * Sketched l2 regressor using sketch-and-solve when sketch fits into a
 * single node.
 *
 * This implementation assumes RhsType is the same as MatrixType since
 * we apply the same sketch to both the Matrix and Rhs.
 */
template <typename ValueType,
          typename MatrixType,
          template <typename, typename> class TransformType,
          typename ExactAlgTag>
class sketched_regressor_t<l2_tag,
                           MatrixType,
                           MatrixType,
                           elem::Matrix<ValueType>,
                           TransformType,
                           ExactAlgTag,
                           sketch_and_solve_tag> {

    typedef elem::Matrix<ValueType> sketch_type;

    typedef regression_problem_t<l2_tag, MatrixType> problem_type;
    typedef regression_problem_t<l2_tag, sketch_type> sketched_problem_type;

    typedef exact_regressor_t<l2_tag,
                              sketch_type,
                              sketch_type,
                              ExactAlgTag> underlying_regressor_type;

    typedef TransformType<MatrixType, sketch_type> sketch_transform_type;

private:
    const int _my_rank;
    const int _sketch_size;
    sketch_transform_type *_sketch_transform;
    underlying_regressor_type  *_underlying_regressor;

public:
    sketched_regressor_t(const problem_type &problem, int sketch_size,
        skylark::base::context_t& context) :
        _my_rank(boost::mpi::communicator().rank()), _sketch_size(sketch_size) {
        // TODO m < n
        _sketch_transform =
            new sketch_transform_type(problem.m, sketch_size, context);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        sketch_type sketch(sketch_size, problem.n);
        _sketch_transform->apply(problem.input_matrix, sketch,
            sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_regressor = new underlying_regressor_type(sketched_problem);
    }

    ~sketched_regressor_t() {
        delete _underlying_regressor;
        delete _sketch_transform;
    }

    void solve(const MatrixType &b, sketch_type &x) {
        sketch_type Sb(_sketch_size, 1);
        _sketch_transform->apply(b, Sb, sketch::columnwise_tag());
        if (_my_rank == 0)
            _underlying_regressor->solve(Sb, x);
    }

    void solve_mulitple(const MatrixType &B, sketch_type &X) {
        sketch_type SB(_sketch_size, B.Width());
        _sketch_transform->apply(SB, SB, sketch::columnwise_tag());
        if (_my_rank == 0)
            _underlying_regressor->solve_mulitple(SB, X);
    }
};

/**
 * Sketched l2 regressor using sketch-and-solve when both input and
 * output are distributed.
 *
 * This implementation assumes RhsType is the same as MatrixType since
 * we apply the same sketch to both the Matrix and Rhs. We further
 * make the simplifying assumption that sketch distribution
 * is the same.
 */
template <typename VT,
          elem::Distribution CD, elem::Distribution RD,
          template <typename, typename> class TransformType,
          typename ExactAlgTag>
class sketched_regressor_t<l2_tag,
                           elem::DistMatrix<VT, CD, RD>,
                           elem::DistMatrix<VT, CD, RD>,
                           elem::DistMatrix<VT, CD, RD>,
                           TransformType,
                           ExactAlgTag,
                           sketch_and_solve_tag> {

    typedef elem::DistMatrix<VT, CD, RD> matrix_type;
    typedef elem::DistMatrix<VT, CD, RD> sketch_type;

    typedef regression_problem_t<l2_tag, matrix_type> problem_type;
    typedef regression_problem_t<l2_tag, sketch_type> sketched_problem_type;

    typedef exact_regressor_t<l2_tag,
                              sketch_type,
                              sketch_type,
                              ExactAlgTag> underlying_regressor_type;

    typedef TransformType<matrix_type, sketch_type> sketch_transform_type;

private:
    const int _my_rank;
    const int _sketch_size;
    sketch_transform_type *_sketch_transform;
    underlying_regressor_type  *_underlying_regressor;

public:
    sketched_regressor_t(const problem_type &problem, int sketch_size,
        skylark::base::context_t& context) :
        _my_rank(boost::mpi::communicator().rank()), _sketch_size(sketch_size) {
        // TODO m < n
        _sketch_transform =
            new sketch_transform_type(problem.m, sketch_size, context);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        sketch_type sketch(sketch_size, problem.n);
        _sketch_transform->apply(problem.input_matrix, sketch,
            sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_regressor = new underlying_regressor_type(sketched_problem);
    }

    ~sketched_regressor_t() {
        delete _underlying_regressor;
        delete _sketch_transform;
    }

    void solve(const matrix_type &b, sketch_type &x) {
        // TODO For DistMatrix this will allocate on DefaultGrid
        //      MIGHT BE VERY WRONG (grid is different).
        sketch_type Sb(_sketch_size, 1);
        _sketch_transform->apply(b, Sb, sketch::columnwise_tag());
        _underlying_regressor->solve(Sb, x);
    }

    void solve_mulitple(const matrix_type &B, sketch_type &X) {
        // TODO For DistMatrix this will allocate on DefaultGrid...
        //      MIGHT BE VERY WRONG (grid is different).
        sketch_type SB(_sketch_size, B.Width());
        _sketch_transform->apply(SB, SB, sketch::columnwise_tag());
        _underlying_regressor->solve_mulitple(SB, X);
    }
};

} // namespace algorithms
} // namespace skylark
#endif // SKETCHED_REGRESSOR_ELEMENTAL_HPP
