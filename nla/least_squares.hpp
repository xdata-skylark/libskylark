#ifndef SKYLARK_LEAST_SQUARES_HPP
#define SKYLARK_LEAST_SQUARES_HPP

#include <elemental.hpp>
#include "../algorithms/regression/regression.hpp"
#include "../base/exception.hpp"

namespace skylark { namespace nla {

template<typename T>
void ApproximateLeastSquares(elem::Orientation orientation,
    const elem::Matrix<T>& A, const elem::Matrix<T>& B, elem::Matrix<T>& X, 
    base::context_t& context, int sketch_size = -1) {

    if (orientation != elem::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<elem::Matrix<double>,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype, elem::Matrix<double>, elem::Matrix<double>,
        algorithms::linear_tag,
        elem::Matrix<double>,
        elem::Matrix<double>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

template<typename T, elem::Distribution CA, elem::Distribution RA,
         elem::Distribution CB, elem::Distribution RB, elem::Distribution CX, 
         elem::Distribution RX>
void ApproximateLeastSquares(elem::Orientation orientation,
    const elem::DistMatrix<T, CA, RA>& A, const elem::DistMatrix<T, CB, RB>& B,
    elem::DistMatrix<T, CX, RX>& X, base::context_t& context,
    int sketch_size = -1) {

    if (orientation != elem::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<elem::DistMatrix<T, CA, RA>,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype,
        elem::DistMatrix<T, CB, RB>,
        elem::DistMatrix<T, CX, RX>,
        algorithms::linear_tag,
        elem::DistMatrix<T, elem::STAR, elem::STAR>,
        elem::DistMatrix<T, elem::STAR, elem::STAR>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

/*
template<typename AT, typename BT, typename XT>
void ApproximateLeastSquares(elem::Orientation orientation, const AT& A, const BT& B,
    XT& X, base::context_t& context, int sketch_size = -1) {

    if (orientation != elem::NORMAL) 
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<AT,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype, BT, XT,
        algorithms::linear_tag,
        elem::DistMatrix<double, elem::STAR, elem::STAR>,
        elem::DistMatrix<double, elem::STAR, elem::STAR>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}
*/

template<typename AT, typename BT, typename XT>
void FastLeastSquares(elem::Orientation orientation, const AT& A, const BT& B,
    XT& X, base::context_t& context) {

    if (orientation != elem::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for FastLeastSquares"));

    typedef algorithms::regression_problem_t<AT,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::accelerated_regression_solver_t<ptype, BT, XT,
                                    algorithms::blendenpik_tag<
                                        algorithms::qr_precond_tag> >
        solver(problem, context);
    solver.solve(B, X);
}


} }
#endif
