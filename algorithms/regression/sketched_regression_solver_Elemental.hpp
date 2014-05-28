#ifndef SKYLARK_SKETCHED_REGRESSION_SOLVER_ELEMENTAL_HPP
#define SKYLARK_SKETCHED_REGRESSION_SOLVER_ELEMENTAL_HPP

#include <boost/mpi.hpp>
#include <elemental.hpp>

#include "../../base/context.hpp"
#include "regression_problem.hpp"
#include "../../sketch/sketch.hpp"
#include "../../utility/typer.hpp"

namespace skylark {
namespace algorithms {

/**
 * Generic sketched regression solver using sketch-and-solve when sketch fits
 * int single node.
 */
template <
    typename RegressionType,
    typename PenaltyType,
    typename RegularizationType,
    typename InputType,
    typename RhsType,
    typename SolType,
    typename SketchedRegressionType,
    template <typename, typename> class TransformType,
    typename ExactAlgTag>
class sketched_regression_solver_t<
    regression_problem_t<InputType,
                         RegressionType, PenaltyType, RegularizationType>,
    RhsType,
    SolType,
    SketchedRegressionType,
    elem::Matrix<
        typename utility::typer_t<InputType>::value_type >,
    elem::Matrix<
        typename utility::typer_t<InputType>::value_type >,
    TransformType,
    ExactAlgTag> {

public:

    typedef typename utility::typer_t<InputType>::value_type value_type;

    typedef elem::Matrix<value_type> sketch_type;
    typedef elem::Matrix<value_type> sketch_rhs_type;
    typedef InputType matrix_type;
    typedef RhsType rhs_type;
    typedef SolType sol_type;

    typedef RegressionType regression_type;
    typedef PenaltyType penalty_type;
    typedef RegularizationType regularization_type;
    typedef SketchedRegressionType sketched_regression_type;

    typedef regression_problem_t<matrix_type,
                                 regression_type, penalty_type,
                                 regularization_type> problem_type;
    typedef regression_problem_t<sketch_type,
                                 sketched_regression_type, penalty_type,
                                 regularization_type> sketched_problem_type;


    typedef regression_solver_t<sketched_problem_type,
                              sketch_rhs_type,
                              sol_type,
                              ExactAlgTag> underlying_solver_type;

private:
    typedef typename TransformType<matrix_type, sketch_type>::data_type
    transform_data_type;

    const int _my_rank;
    const int _sketch_size;
    const transform_data_type _sketch;
    const underlying_solver_type  *_underlying_solver;

public:
    sketched_regression_solver_t(const problem_type& problem, int sketch_size,
        base::context_t& context) :
        _my_rank(utility::get_communicator(problem.input_matrix)),
        _sketch_size(sketch_size),
        _sketch(problem.m, sketch_size, context) {

        // TODO m < n
        TransformType<matrix_type, sketch_type> S(_sketch);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        sketch_type sketch(sketch_size, problem.n);
        S.apply(problem.input_matrix, sketch, sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_solver = new underlying_solver_type(sketched_problem);
    }

    ~sketched_regression_solver_t() {
        delete _underlying_solver;
    }

    void solve(const rhs_type& b, sol_type& x) {
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type Sb(_sketch_size, 1);
        S.apply(b, Sb, sketch::columnwise_tag());
        if (_my_rank == 0)
            _underlying_solver->solve(Sb, x);
    }

    void solve_mulitple(const rhs_type& B, sol_type& X) {
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type SB(_sketch_size, B.Width());
        S.apply(SB, SB, sketch::columnwise_tag());
        if (_my_rank == 0)
            _underlying_solver->solve_mulitple(SB, X);
    }
};

/**
 * Generic sketched regressor using sketch-and-solve when sketch is distributed.
 */
template <
    typename RegressionType,
    typename PenaltyType,
    typename RegularizationType,
    typename InputType,
    typename RhsType,
    typename SolType,
    typename SketchedRegressionType,
    elem::Distribution CD, elem::Distribution RD,
    template <typename, typename> class TransformType,
    typename ExactAlgTag>
class sketched_regression_solver_t<
    regression_problem_t<InputType,
                         RegressionType, PenaltyType, RegularizationType>,
    RhsType,
    SolType,
    SketchedRegressionType,
    elem::DistMatrix<
        typename utility::typer_t<InputType>::value_type,
        CD, RD >,
   elem::DistMatrix<
        typename utility::typer_t<InputType>::value_type,
        CD, RD >,
    TransformType,
    ExactAlgTag> {

public:

    typedef typename utility::typer_t<InputType>::value_type value_type;

    typedef elem::DistMatrix<value_type, CD, RD> sketch_type;
    typedef elem::DistMatrix<value_type, CD, RD> sketch_rhs_type;
    typedef InputType matrix_type;
    typedef RhsType rhs_type;
    typedef SolType sol_type;

    typedef RegressionType regression_type;
    typedef PenaltyType penalty_type;
    typedef RegularizationType regularization_type;
    typedef SketchedRegressionType sketched_regression_type;

    typedef regression_problem_t<matrix_type,
                                 regression_type, penalty_type,
                                 regularization_type> problem_type;
    typedef regression_problem_t<sketch_type,
                                 sketched_regression_type, penalty_type,
                                 regularization_type> sketched_problem_type;

    typedef regression_solver_t<sketched_problem_type,
                                sketch_rhs_type,
                                sol_type,
                                ExactAlgTag> underlying_solver_type;

private:
    typedef typename TransformType<matrix_type, sketch_type>::data_type
    transform_data_type;

    const int _sketch_size;
    const transform_data_type _sketch;
    const underlying_solver_type  *_underlying_solver;

public:
    sketched_regression_solver_t(const problem_type& problem, int sketch_size,
        base::context_t& context) :
        _sketch_size(sketch_size),
        _sketch(problem.m, sketch_size, context) {

        // TODO m < n
        TransformType<matrix_type, sketch_type> S(_sketch);
        // TODO For DistMatrix this will allocate on DefaultGrid...
        sketch_type sketch(sketch_size, problem.n);
        S.apply(problem.input_matrix, sketch, sketch::columnwise_tag());
        sketched_problem_type sketched_problem(sketch_size, problem.n, sketch);
        _underlying_solver = new underlying_solver_type(sketched_problem);
    }

    ~sketched_regression_solver_t() {
        delete _underlying_solver;
    }

    void solve(const rhs_type& b, sol_type& x) {
        // TODO For DistMatrix this will allocate on DefaultGrid
        //      MIGHT BE VERY WRONG (grid is different).
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type Sb(_sketch_size, 1);
        S.apply(b, Sb, sketch::columnwise_tag());
        _underlying_solver->solve(Sb, x);
    }

    void solve_mulitple(const rhs_type& B, sol_type& X) {
        // TODO For DistMatrix this will allocate on DefaultGrid...
        //      MIGHT BE VERY WRONG (grid is different).
        TransformType<rhs_type, sketch_type> S(_sketch);
        sketch_type SB(_sketch_size, B.Width());
        S.apply(SB, SB, sketch::columnwise_tag());
        _underlying_solver->solve_mulitple(SB, X);
    }
};

} } // namespace skylark::algorithms

#endif // SKYLARK_SKETCHED_REGRESSION_SOLVER_ELEMENTAL_HPP
