#ifndef SKYLARK_LEAST_SQUARES_HPP
#define SKYLARK_LEAST_SQUARES_HPP

#include <El.hpp>
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
 * \param orientation If El::NORMAL will approximate 
 *                    argmin_X ||A * X - B||_F
 *                    If El::ADJOINT will approximate (NOT YET SUPPORTED)
 *                    argmin_X ||A^H * X - B||_F
 * \param A input matrix
 * \param B right-hand side
 * \param X solution matrix
 * \param sketch_size Sketch size to use. Higher values will produce better 
 *                    approximations. Default is 4 * Width(A).
 */
template<typename T>
void ApproximateLeastSquares(El::Orientation orientation,
    const El::Matrix<T>& A, const El::Matrix<T>& B, El::Matrix<T>& X, 
    base::context_t& context, int sketch_size = -1) {

    if (orientation != El::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::nla_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<El::Matrix<double>,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype, El::Matrix<T>, El::Matrix<T>,
        algorithms::linear_tag,
        El::Matrix<T>,
        El::Matrix<T>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

/**
 * See documentation for local matrix variant.
 */
template<typename T, El::Distribution U, El::Distribution V>
void ApproximateLeastSquares(El::Orientation orientation,
    const El::DistMatrix<T, U, V>& A,
    const El::DistMatrix<T, U, V>& B,
    El::DistMatrix<T, U, V>& X,
    base::context_t& context, int sketch_size = -1) {

    if (orientation != El::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::nla_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<El::DistMatrix<T, U, V>,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype,
        El::DistMatrix<T, U, V>,
        El::DistMatrix<T, U, V>,
        algorithms::linear_tag,
        El::DistMatrix<T, U, V>,
        El::DistMatrix<T, U, V>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

/**
 * See documentation for local matrix variant.
 */
template<typename T>
void ApproximateLeastSquares(El::Orientation orientation,
    const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& X,
    base::context_t& context, int sketch_size = -1) {

    if (orientation != El::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::nla_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<El::DistMatrix<T, El::VC, El::STAR>,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype,
        El::DistMatrix<T, El::VC, El::STAR>,
        El::DistMatrix<T, El::STAR, El::STAR>,
        algorithms::linear_tag,
        El::DistMatrix<T, El::STAR, El::STAR>,
        El::DistMatrix<T, El::STAR, El::STAR>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

/**
 * See documentation for local matrix variant.
 */
template<typename T>
void ApproximateLeastSquares(El::Orientation orientation,
    const El::DistMatrix<T, El::VR, El::STAR>& A,
    const El::DistMatrix<T, El::VR, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& X,
    base::context_t& context, int sketch_size = -1) {

    if (orientation != El::NORMAL)
        SKYLARK_THROW_EXCEPTION (
          base::nla_exception()
              << base::error_msg(
                 "Only NORMAL orientation is supported for ApproximateLeastSquares"));

    if (sketch_size == -1)
        sketch_size = 4 * base::Width(A);

    typedef algorithms::regression_problem_t<El::DistMatrix<T, El::VR, El::STAR>,
                                             algorithms::linear_tag,
                                             algorithms::l2_tag,
                                             algorithms::no_reg_tag> ptype;
    ptype problem(base::Height(A), base::Width(A), A);

    algorithms::sketched_regression_solver_t<
        ptype,
        El::DistMatrix<T, El::VR, El::STAR>,
        El::DistMatrix<T, El::STAR, El::STAR>,
        algorithms::linear_tag,
        El::DistMatrix<T, El::STAR, El::STAR>,
        El::DistMatrix<T, El::STAR, El::STAR>,
        sketch::FJLT_t,
        algorithms::qr_l2_solver_tag> solver(problem, sketch_size, context);

    solver.solve(B, X);
}

/**
 * Solve the linear least-squares problem
 *
 *   argmin_X ||A * X - B||_F
 *
 * using a sketching-accelerated algorithm. This algorithm uses sketching
 * to build a preconditioner, and then uses the preconditioner in an iterative
 * method. While technically the solution found is approximate (due to the use
 * of an iterative method), the threshold is set close to machine precision
 * so the solution's accuracy is close to the full accuracy possible on a
 * machine.
 *
 * The algorithm implemented is the one described in:
 *
 * Haim Avron, Petar Maymounkov, and Sivan Toledo
 * Blendenpik: Supercharging LAPACK's Least-Squares Solver
 * SIAM Journal on Scientific Computing 32(3), 1217-1236, 2010
 *
 * Note: it is assume that a 4*Width(A)^2 matrix can fit in memory
 * of a single node.
 *
 * \param orientation If El::NORMAL will approximate 
 *                    argmin_X ||A * X - B||_F
 *                    If El::ADJOINT will approximate (NOT YET SUPPORTED)
 *                    argmin_X ||A^H * X - B||_F
 * \param A input matrix
 * \param B right-hand side
 * \param X solution matrix
 */
template<typename AT, typename BT, typename XT>
void FastLeastSquares(El::Orientation orientation, const AT& A, const BT& B,
    XT& X, base::context_t& context) {

    if (orientation != El::NORMAL)
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
