#ifndef SKYlARK_EXACT_REGRESSOR_KRYLOV_HPP
#define SKYLARK_EXACT_REGRESSOR_KRYLOV_HPP

#include "../../base/query.hpp"
#include "../../utility/typer.hpp"
#include "../../nla/LSQR.hpp"

namespace skylark { namespace algorithms {

/**
 * Exact l2 linear regressor. We have not specialized this for any type of matrix
 * because it accepts any matrix type and only works if the required base
 * functions have been implemented.
 */
template <typename MatrixType,
          typename RhsType,
          typename SolType,
          typename KrylovMethod>
struct exact_regressor_t<
    regression_problem_t<MatrixType, linear_tag, l2_tag, no_reg_tag>,
    RhsType,
    SolType,
    iterative_l2_solver_tag<KrylovMethod> > {

    typedef typename utility::typer_t<MatrixType>::value_type value_type;

    typedef MatrixType matrix_type;
    typedef RhsType rhs_type;
    typedef SolType sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

    const int m;
    const int n;
    const matrix_type& A;

    exact_regressor_t (const problem_type& problem) :
        m(problem.m), n(problem.n), A(problem.input_matrix)
    { /* Check if m<n? */ }

    /** * A solve implementation that uses LSQR */
    int solve_impl (const rhs_type& b,
        sol_type& x,
        skylark::nla::iter_params_t &params,
        lsqr_tag) {

        return LSQR(A, b, x, params);
    }

    int solve(const rhs_type& b,
        sol_type& x,
        skylark::nla::iter_params_t params = skylark::nla::iter_params_t()) {


        if (m != base::Height(b)) { /* error */ return -1; }
        if (n != base::Height(x)) { /* error */ return -1; }
        if (base::Width(b) != base::Width(x)) { /* error */ return -1; }

        /**
         * Solve using the right iterative solver that is specified for us.
         * The reason we are using tag-based dispatching to different solve
         * implementations is because different iterative solvers might have
         * different initialization requirements. This technique gives us an
         * opportunity to handle each iterative algorithm in a different func.
         */
        return solve_impl (b, x, params, KrylovMethod());
    }
};

} } // namespace skylark::algorithms

#endif // SKYLARK_EXACT_REGRESSOR_KRYLOV_HPP
