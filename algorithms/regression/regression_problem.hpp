#ifndef REGRESSION_PROBLEM_HPP
#define REGRESSION_PROBLEM_HPP

namespace skylark {
namespace algorithms {

/// Tag for specifying a L2 regression problem.
struct l2_tag {};

/// Tag for specifying a L1 regression problem.
struct l1_tag {};

/// Tag for specifying a Lp regression problem, where p = dem / num.
template<int dem, int num>
struct lp_tag {};

/**
 * Describes a regression problem on a matrix (right hand side supplied when
 * the solver is invoked).
 *
 * @tparam RegressionType Tag specificying regression type, like l2_tag.
 * @tparam InputMatrixType Specifies input matrix type.
 */
template <typename RegressionType, typename InputMatrixType>
struct regression_problem_t {

    regression_problem_t(int m, int n, const InputMatrixType &input_matrix) :
        m(m), n(n), input_matrix(input_matrix) {
        // TODO verify size of input_matrix
    }

    const int m;                          ///< Number of constraints.
    const int n;                          ///< Number of variables.
    const InputMatrixType &input_matrix;  ///< Input matrix.

    // TODO add regularization option(s) ?
    // TODO add additive model option ?
    // TODO regularizers, specialized parameters (based on RegressionType)
};

} // namespace sketch
} // namespace skylark

#endif // REGRESSION_PROBLEM_HPP
