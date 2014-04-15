#ifndef SKYLARK_REGRESSION_PROBLEM_HPP
#define SKYLARK_REGRESSION_PROBLEM_HPP

namespace skylark {
namespace algorithms {

///////////// Tags for regression types

/// Tag for specifying a linear regression.
struct linear_tag {};

/// Tag for specifying a polynomial regression.
struct polynomial_tag {};

/// Tag for specifying a kernel regression.
struct kernel_tag {};

// TODO more

///////////// Tags for penalty functions

/// Tag for specifying a L2 norm.
struct l2_tag {};

/// Tag for specifying a L1 norm.
struct l1_tag {};

/// Tag for specifying a Lp norm, where p = dem / num.
template<int dem, int num>
struct lp_tag {};

/**
 * Describes a regression problem on a matrix (right hand side supplied when
 * the solver is invoked). Actual content is specialized based on the input 
 * template tags
 *
 * @tparam RegressionType Tag specificying regression type, like linear_tag.
 * @tparam PenaltyType Tag specificying the penalty function, like l2_tag.
 * @tparam InputMatrixType Specifies input matrix type.
 */
template<
    typename InputMatrixType,
    typename RegressionType,
    typename PenaltyType>
struct regression_problem_t {

};

/**
 * Specialization for linear regression.
 */
template <typename InputMatrixType,
          typename PenaltyType>
struct regression_problem_t<InputMatrixType, linear_tag, PenaltyType> {

    regression_problem_t(int m, int n, const InputMatrixType &input_matrix) :
        m(m), n(n), input_matrix(input_matrix) {
        // TODO verify size of input_matrix
    }

    const int m;                          ///< Number of constraints.
    const int n;                          ///< Number of variables.
    const InputMatrixType& input_matrix;  ///< Input matrix.

    // TODO add regularization option(s) ?
    // TODO regularizers, specialized parameters (based on RegressionType)
};

} // namespace algorithms
} // namespace skylark

#endif // SKYLARK_REGRESSION_PROBLEM_HPP
