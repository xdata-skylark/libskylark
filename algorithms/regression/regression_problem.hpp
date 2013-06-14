#ifndef REGRESSION_PROBLEM_HPP
#define REGRESSION_PROBLEM_HPP

namespace skylark {
namespace algorithms {

struct l2_tag {};
struct l1_tag {};
template<int dem, int num>
struct lp_tag {};

/**
 * Describes a regression problem on a matrix (right hand side supplied when
 * the solver is invoked).
 *
 * TODO add regularization option
 * TODO add additive model option
 */
template <typename RegressionType, typename InputMatrixType>
struct regression_problem_t {

    regression_problem_t(int m, int n, const InputMatrixType &input_matrix) :
        m(m), n(n), input_matrix(input_matrix) {
        // TODO verify size of input_matrix
    }

    const int m;                          /// Number of constraints.
    const int n;                          /// Number of variables.
    const InputMatrixType &input_matrix;

    // TODO regularizers, specialized parameters (based on RegressionType)
};

} // namespace sketch
} // namespace skylark

#endif // REGRESSION_PROBLEM_HPP
