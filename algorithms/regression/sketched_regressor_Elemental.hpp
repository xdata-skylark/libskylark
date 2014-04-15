#ifndef SKYLARK_SKETCHED_REGRESSOR_ELEMENTAL_HPP
#define SKYLARK_SKETCHED_REGRESSOR_ELEMENTAL_HPP

#include <boost/mpi.hpp>
#include <elemental.hpp>

#include "../../base/context.hpp"
#include "regression_problem.hpp"
#include "../../sketch/sketch.hpp"
#include "utility/typer.hpp"

namespace skylark {
namespace algorithms {

/**
 * Generic sketched regressor using sketch-and-solve when sketch fits into a
 * single node.
 */
template <
    typename RegressionType,
    typename InputType,
    typename RhsType,
    template <typename, typename> class TransformType,
    typename ExactAlgTag>
class sketched_regressor_t<RegressionType,
                           InputType,
                           RhsType,
                           elem::Matrix<
                               typename utility::typer_t<InputType>::value_type >,
                           TransformType,
                           ExactAlgTag,
                           sketch_and_solve_tag> {

    typedef typename utility::typer_t<InputType>::value_type value_type;

    typedef InputType matrix_type;
    typedef RhsType rhs_type;
    typedef elem::Matrix<value_type> sketch_type;

    typedef RegressionType regression_type;

    typedef regression_problem_t<regression_type,
                                 matrix_type> problem_type;
    typedef regression_problem_t<regression_type,
                                 sketch_type> sketched_problem_type;

    typedef exact_regressor_t<regression_type,
                              sketch_type,
                              sketch_type,
                              ExactAlgTag> underlying_regressor_type;

private:
    typedef typename TransformType<matrix_type, sketch_type>::data_type
    transform_data_type;

    const int _my_rank;
    const int _sketch_size;
    const transform_data_type _sketch;
    const underlying_regressor_type  *_underlying_regressor;

public:
    sketched_regressor_t(const problem_type& problem, int sketch_size,
        base::context_t& context) :
        _my_rank(utility::get_communicator(problem.input_matrix)),
        _sketch_size(sketch_size),
        _sketch(sketch_size, problem.n, context) {

        // TODO m < n
        TransformType<matrix_type, sketch_type> S(_sketch);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        sketch_type sketch(sketch_size, problem.n);
        S.apply(problem.input_matrix, sketch, sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_regressor = new underlying_regressor_type(sketched_problem);
    }

    ~sketched_regressor_t() {
        delete _underlying_regressor;
    }

    void solve(const matrix_type& b, sketch_type& x) {
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type Sb(_sketch_size, 1);
        S.apply(b, Sb, sketch::columnwise_tag());
        if (_my_rank == 0)
            _underlying_regressor->solve(Sb, x);
    }

    void solve_mulitple(const matrix_type& B, sketch_type& X) {
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type SB(_sketch_size, B.Width());
        S.apply(SB, SB, sketch::columnwise_tag());
        if (_my_rank == 0)
            _underlying_regressor->solve_mulitple(SB, X);
    }
};

/**
 * Generic sketched regressor using sketch-and-solve when sketch is distributed.
 */
template <
    typename RegressionType,
    typename InputType,
    typename RhsType,
    elem::Distribution CD, elem::Distribution RD,
    template <typename, typename> class TransformType,
    typename ExactAlgTag>
class sketched_regressor_t<RegressionType,
                           InputType,
                           RhsType,
                           elem::DistMatrix<
                               typename utility::typer_t<InputType>::value_type,
                               CD, RD >,
                           TransformType,
                           ExactAlgTag,
                           sketch_and_solve_tag> {

    typedef typename utility::typer_t<InputType>::value_type value_type;

    typedef InputType matrix_type;
    typedef RhsType rhs_type;
    typedef elem::DistMatrix<value_type, CD, RD> sketch_type;

    typedef RegressionType regression_type;

    typedef regression_problem_t<regression_type,
                                 matrix_type> problem_type;
    typedef regression_problem_t<regression_type,
                                 sketch_type> sketched_problem_type;

    typedef exact_regressor_t<regression_type,
                              sketch_type,
                              sketch_type,
                              ExactAlgTag> underlying_regressor_type;

private:
    typedef typename TransformType<matrix_type, sketch_type>::data_type
    transform_data_type;

    const int _sketch_size;
    const transform_data_type _sketch;
    const underlying_regressor_type  *_underlying_regressor;

public:
    sketched_regressor_t(const problem_type& problem, int sketch_size,
        base::context_t& context) :
        _sketch_size(sketch_size),
        _sketch(sketch_size, problem.n, context) {

        // TODO m < n
        TransformType<matrix_type, sketch_type> S(_sketch);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        sketch_type sketch(sketch_size, problem.n);
        S.apply(problem.input_matrix, sketch, sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_regressor = new underlying_regressor_type(sketched_problem);
    }

    ~sketched_regressor_t() {
        delete _underlying_regressor;
    }

    void solve(const matrix_type& b, sketch_type& x) {
        // TODO For DistMatrix this will allocate on DefaultGrid
        //      MIGHT BE VERY WRONG (grid is different).
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type Sb(_sketch_size, 1);
        S.apply(b, Sb, sketch::columnwise_tag());
        _underlying_regressor->solve(Sb, x);
    }

    void solve_mulitple(const matrix_type& B, sketch_type& X) {
        // TODO For DistMatrix this will allocate on DefaultGrid...
        //      MIGHT BE VERY WRONG (grid is different).
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type SB(_sketch_size, B.Width());
        S.apply(SB, SB, sketch::columnwise_tag());
        _underlying_regressor->solve_mulitple(SB, X);
    }
};

} } // namespace skylark::algorithms

#endif // SKYLARK_SKETCHED_REGRESSOR_ELEMENTAL_HPP
