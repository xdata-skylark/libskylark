#ifndef EXACT_REGRESSOR_KRYLOV_HPP
#define EXACT_REGRESSOR_KRYLOV_HPP

#include "../../nla/lsqr.hpp"

namespace skylark { namespace algorithms {

/**
 * Exact l2 linear regressor. We have not specialized this for any type of matrix
 * because it accepts any matrix type and only functions if iter_ops_t has
 * been specialized for that particular matrix type and that particular
 * multi-vector type. Currently, the accepted ones include:
 * (1) <DistMatrix<NT, CDT, RDT>, DistMatrix<NT, CDT, RDT> >
 * (2) <SpParMat<IT, NT, SpDCCols<IT,NT> >, FullyDistMultiVec>
 */
template <typename MatrixType,
          typename MultiVectorType,
          typename KrylovMethod>
struct exact_regressor_t<
    regression_problem_t<MatrixType, linear_tag, l2_tag, no_reg_tag>,
    MultiVectorType,
    iterative_l2_solver_tag<KrylovMethod> > {

    typedef MatrixType matrix_type;
    typedef MultiVectorType rhs_type;
    typedef MultiVectorType multivec_type;
    typedef skylark::nla::iter_solver_op_t<matrix_type, multivec_type> iter_ops_t;
    typedef typename iter_ops_t::value_type value_type;
    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

    const int m;
    const int n;
    const matrix_type& A;

    exact_regressor_t (const problem_type& problem) :
        m(problem.m), n(problem.n), A(problem.input_matrix)
    { /* Check if m<n? */ }

    /** * A solve implementation that uses LSQR */
    void solve_impl (const rhs_type& b,
        rhs_type& x,
        skylark::nla::iter_params_t params,
        lsqr_tag) {
        /** Call the LSQR solver */
        skylark::nla::lsqr_t<matrix_type, rhs_type>::apply (A, b, x, params);
    }

    void solve(const rhs_type& b,
        rhs_type& x,
        skylark::nla::iter_params_t params) {
        int b_m, b_n, x_m, x_n;
        iter_ops_t::get_dim (b, b_m, b_n);
        iter_ops_t::get_dim (x, x_m, x_n);

        if (m != b_m) { /* error */ return; }
        if (n != x_m) { /* error */ return; }
        if (b_n != x_n) { /* error */ return; }

        /**
         * Solve using the right iterative solver that is specified for us. 
         * The reason we are using tag-based dispatching to different solve
         * implementations is because different iterative solvers might have
         * different initialization requirements. This technique gives us an
         * opportunity to handle each iterative algorithm in a different func.
         */
        solve_impl (b, x, params, KrylovMethod());
    }
};

} } // namespace skylark::algorithms

#endif // EXACT_REGRESSOR_KRYLOV_HPP
