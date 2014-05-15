#ifndef SKYlARK_LINEARL2_REGRESSION_SOLVER_KRYLOV_HPP
#define SKYLARK_LINEARL2_REGRESSION_SOLVER_KRYLOV_HPP

#include "../../base/query.hpp"
#include "../../utility/typer.hpp"
#include "../../nla/LSQR.hpp"

namespace skylark { namespace algorithms {

/**
 * l2 linear regression solver using Krylov method. We have not specialized this
 * for any type of matrix because it accepts any matrix type and only works if 
 * the required base functions have been implemented.
 */
template <typename MatrixType,
          typename RhsType,
          typename SolType,
          typename KrylovMethod>
struct regression_solver_t<
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

private:
    const nla::id_precond_t<sol_type> _id_precond_obj;

public:
    const int m;
    const int n;
    const matrix_type& A;
    const nla::precond_t<sol_type>& R;
    const nla::iter_params_t iter_params;

    regression_solver_t (const problem_type& problem,
        nla::iter_params_t iter_params = nla::iter_params_t()) :
        m(problem.m), n(problem.n), A(problem.input_matrix),
        R(_id_precond_obj), iter_params(iter_params)
    { /* Check if m<n? */ }

    regression_solver_t (const problem_type& problem,
        const nla::precond_t<sol_type>& R,
        nla::iter_params_t iter_params = nla::iter_params_t()) :
        m(problem.m), n(problem.n), A(problem.input_matrix), R(R),
        iter_params(iter_params)
    { /* Check if m<n? */ }



    int solve(const rhs_type& b, sol_type& x) {

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
        return solve_impl (b, x, KrylovMethod());
    }

private:
    /** * A solve implementation that uses LSQR */
    int solve_impl (const rhs_type& b, sol_type& x, lsqr_tag) {

        return LSQR(A, b, x, iter_params, R);
    }
};

} } // namespace skylark::algorithms

#endif // SKYLARK_LINEARL2_REGRESSION_SOLVER_KRYLOV_HPP
