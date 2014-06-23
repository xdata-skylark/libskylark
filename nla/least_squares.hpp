#ifndef SKYLARK_LEAST_SQUARES_HPP
#define SKYLARK_LEAST_SQUARES_HPP

#include <elemental.hpp>
#include "../algorithms/regression/regression.hpp"
#include "../base/exception.hpp"

namespace skylark { namespace nla {

/**
 * Solve the linear least-squares problem
 *
 *   argmin_X ||A * X - B||_F
 *
 * approximately using sketching. This algorithm uses the sketch-and-solve
 * strategy. The algorithm implemented is the one described in:
 *
 * P. Drineas, M. W. Mahoney, S. Muthukrishnan, and T. Sarlos
 * Faster Least Squares Approximation
 * Numerische Mathematik, 117, 219-249 (2011).
 *
 * although we allow the user to set the sketch size. The default value is also
 * much lower than the one advocated in that paper, so use default options with
 * care.
 *
 * Note: it is assume that a sketch_size x Width(A) matrix can fit in memory
 * of a single node.
 *
 * \param orientation If elem::NORMAL will approximate 
 *                    argmin_X ||A * X - B||_F
 *                    If elem::ADJOINT will approximate (NOT YET SUPPORTED)
 *                    argmin_X ||A^H * X - B||_F
 * \param A input matrix
 * \param B right-hand side
 * \param X solution matrix
 * \param sketch_size Sketch size to use. Higher values will produce better 
 *                    approximations. Default is 4 * Width(A).
 */
template<typename T>
void ApproximateLeastSquares(elem::Orientation orientation,
    const elem::Matrix<T>& A, const elem::Matrix<T>& B, elem::Matrix<T>& X, 
    base::context_t& context, int sketch_size = -1) {

    if (orientation != elem::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::nla_exception()
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
        ptype, elem::Matrix<T>, elem::Matrix<T>,
        algorithms::linear_tag,
        elem::Matrix<T>,
        elem::Matrix<T>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

/**
 * Solve the linear least-squares problem
 *
 *   argmin_X ||A * X - B||_F
 *
 * approximately using sketching. This algorithm uses the sketch-and-solve
 * strategy. The algorithm implemented is the one described in:
 *
 * P. Drineas, M. W. Mahoney, S. Muthukrishnan, and T. Sarlos
 * Faster Least Squares Approximation
 * Numerische Mathematik, 117, 219-249 (2011).
 *
 * although we allow the user to set the sketch size. The default value is also
 * much lower than the one advocated in that paper, so use default options with
 * care.
 *
 * Note: it is assume that a sketch_size x Width(A) matrix can fit in memory
 * of a single node.
 *
 * \param orientation If elem::NORMAL will approximate
 *                    argmin_X ||A * X - B||_F
 *                    If elem::ADJOINT will approximate (NOT YET SUPPORTED)
 *                    argmin_X ||A^H * X - B||_F
 * \param A input matrix
 * \param B right-hand side
 * \param X solution matrix
 * \param sketch_size Sketch size to use. Higher values will produce better
 *                    approximations. Default is 4 * Width(A).
 */
template<typename T, elem::Distribution CA, elem::Distribution RA,
         elem::Distribution CB, elem::Distribution RB, elem::Distribution CX,
         elem::Distribution RX>
void ApproximateLeastSquares(elem::Orientation orientation,
    const elem::DistMatrix<T, CA, RA>& A, const elem::DistMatrix<T, CB, RB>& B,
    elem::DistMatrix<T, CX, RX>& X, base::context_t& context,
    int sketch_size = -1) {

    if (orientation != elem::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::nla_exception()
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

/**
 * Solve the linear least-squares problem
 *
 *   argmin_X ||A * X - B||_F
 *
 * using a sketching-accelerated sketching. This algorithm uses sketching
 * to build a preconditioner, and then uses the preconditioner in an interative
 * method. While technically the solution found is approximate (due to the use
 * of an iterative method), the threshold is set close to machine precision
 * so the solution enjoys close to the full accuracy possible on a machine.
 *
 * The algorithm implemented is the one described in:
 *
 * Haim Avron, Petar Maymounkov, and Sivan Toledo
 * Blendenpik: Supercharging LAPACK's Least-Squares Solver
 * SIAM Journal on Scientific Computing 32(3), 1217-1236, 2010
 *
 * Note: it is assume that a sketch_size x Width(A) matrix can fit in memory
 * of a single node.
 *
 * \param orientation If elem::NORMAL will approximate 
 *                    argmin_X ||A * X - B||_F
 *                    If elem::ADJOINT will approximate (NOT YET SUPPORTED)
 *                    argmin_X ||A^H * X - B||_F
 * \param A input matrix
 * \param B right-hand side
 * \param X solution matrix
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
