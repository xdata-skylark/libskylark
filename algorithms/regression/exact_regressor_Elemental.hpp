#ifndef SKYlARK_EXACT_REGRESSOR_ELEMENTAL_HPP
#define SKYLARK_EXACT_REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "regression_problem.hpp"

namespace skylark {
namespace algorithms {

/**
 * Exact regressor (solves the problem exactly) for L2 linear regression
 * on a dense local matrix.
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class exact_regressor_t<
    regression_problem_t<elem::Matrix<ValueType>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::Matrix<ValueType>,
    elem::Matrix<ValueType>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef elem::Matrix<ValueType> matrix_type;
    typedef elem::Matrix<ValueType> rhs_type;
    typedef elem::Matrix<ValueType> sol_type;

    typedef regression_problem_t<
        elem::Matrix<ValueType>, linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _t;
    matrix_type _R;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    exact_regressor_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n) {
        // TODO n < m
        _QR = problem.input_matrix;
        elem::QR(_QR, _t);
        elem::LockedView(_R, _QR, 0, 0, _n, _n);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve(const rhs_type& B, rhs_type& X) const {
        // TODO error checking
        X = B;
        elem::qr::ApplyQ(elem::LEFT, elem::ADJOINT, _QR, _t, X);
        X.Resize(_n, B.Width());
        elem::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X, true);
    }
};

/**
 * Exact regressor (solves the problem exactly) for L2 linear regression
 * on a [STAR, STAR] DistMatrix (aka all local copies are the same)
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class exact_regressor_t<
    regression_problem_t<elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> matrix_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> rhs_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> sol_type;

    typedef regression_problem_t<
        elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
        linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _t;
    matrix_type _R;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    exact_regressor_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n) {
        // TODO n < m
        _QR = problem.input_matrix;
        elem::QR(_QR.Matrix(), _t.Matrix());
        elem::LockedView(_R.Matrix(), _QR.Matrix(), 0, 0, _n, _n);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve(const rhs_type& B, rhs_type& X) const {
        // TODO error checking
        X = B;
        elem::qr::ApplyQ(elem::LEFT, elem::ADJOINT,
            _QR.LockedMatrix(), _t.LockedMatrix(), X.Matrix());
        X.Resize(_n, B.Width());
        elem::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R.LockedMatrix(), X.Matrix(), true);
    }
};

/**
 * Exact regressor (solves the problem exactly) for L2 linear regrssion
 * on a dense distributed [MC,MR] matrix.
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class exact_regressor_t<
    regression_problem_t<elem::DistMatrix<ValueType>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType>,
    elem::DistMatrix<ValueType>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef elem::DistMatrix<ValueType> matrix_type;
    typedef elem::DistMatrix<ValueType> rhs_type;
    typedef elem::DistMatrix<ValueType> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _R;
    elem::DistMatrix<ValueType, elem::MD, elem::STAR> _t;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    exact_regressor_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n),
        _QR(problem.input_matrix.Grid()), _R(problem.input_matrix.Grid()),
        _t(problem.input_matrix.Grid()) {
        // TODO n < m
        _QR = problem.input_matrix;
        elem::QR(_QR, _t);
        elem::LockedView(_R, _QR, 0, 0, _n, _n);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve (const rhs_type& B, sol_type& X) const {
        // TODO error checking
        X = B;
        elem::qr::ApplyQ(elem::LEFT, elem::ADJOINT, _QR, _t, X);
        X.Resize(_n, B.Width());
        elem::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X);
    }
};

/**
 * Exact regressor (solves the problem exactly) for L2 linear regrssion
 * on a dense distributed [VC/VR, STAR] matrix.
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType, elem::Distribution VD>
class exact_regressor_t<
    regression_problem_t<elem::DistMatrix<ValueType, VD, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef elem::DistMatrix<ValueType, VD, elem::STAR> matrix_type;
    typedef elem::DistMatrix<ValueType, VD, elem::STAR> rhs_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _Q;
    sol_type _R;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    exact_regressor_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n),
        _Q(problem.input_matrix.Grid()), _R(problem.input_matrix.Grid()) {
        // TODO n < m ???
        _Q = problem.input_matrix;
        elem::qr::ExplicitTS(_Q, _R);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve (const rhs_type& B, sol_type& X) const {
        // TODO error checking

        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, _Q, B, X);
        base::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X);
    }

};

} } /** namespace skylark::algorithms */

#endif // SKYLARK_EXACT_REGRESSOR_ELEMENTAL_HPP
