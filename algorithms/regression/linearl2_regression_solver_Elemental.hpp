#ifndef SKYlARK_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP
#define SKYLARK_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP

#include <elemental.hpp>
#include "../../base/base.hpp"

#include "regression_problem.hpp"

namespace skylark {
namespace algorithms {

/**
 * Regression solver for L2 linear regression
 * on a dense local matrix.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class regression_solver_t<
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
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
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
 * Regression solver for L2 linear regression
 * on a [STAR, STAR] DistMatrix (aka all local copies are the same)
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class regression_solver_t<
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
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
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
 * Regression solver for L2 linear regrssion
 * on a dense distributed [MC,MR] matrix.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class regression_solver_t<
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
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
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
 * Regression solver for L2 linear regrssion
 * on a dense distributed [VC/VR, STAR] matrix. This implementation uses
 * a QR based approach.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType, elem::Distribution VD>
class regression_solver_t<
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
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
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

/**
 * Regression solver for L2 linear regrssion
 * on a dense distributed [VC/VR, STAR] matrix. This implementation uses
 * an SVD-based approach.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType, elem::Distribution VD>
class regression_solver_t<
    regression_problem_t<elem::DistMatrix<ValueType, VD, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    svd_l2_solver_tag> {

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
    matrix_type _U;
    sol_type _S, _V;

public:
    /**
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n),
        _U(problem.input_matrix.Grid()), _S(problem.input_matrix.Grid()),
        _V(problem.input_matrix.Grid()) {
        // TODO n < m ???
        _U = problem.input_matrix;
        base::SVD(_U, _S, _V);
        for(int i = 0; i < _S.Height(); i++)
            _S.Set(i, 0, 1 / _S.Get(i, 0));   // TODO handle rank deficiency
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve (const rhs_type& B, sol_type& X) const {
        // TODO error checking
        sol_type UB(X); // Not copying -- just taking grid and size.
        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, _U, B, UB);
        elem::DiagonalScale(elem::LEFT, elem::NORMAL, _S, UB);
        base::Gemm(elem::NORMAL, elem::NORMAL, 1.0, _V, UB, X);
    }

};

/**
 * Regression solver for L2 linear regrssion
 * on a dense distributed [VC/VR, STAR] matrix. This implementation uses
 * an semi-normal equations based approach.
 *
 * NOTE: this solver keeps a reference to the input matrix (needs to be
 * kept in memory).
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType, elem::Distribution VD>
class regression_solver_t<
    regression_problem_t<elem::DistMatrix<ValueType, VD, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    sne_l2_solver_tag> {

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
    const matrix_type& _A;
    sol_type _R;

public:
    /**
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n),
        _A(problem.input_matrix), _R(problem.input_matrix.Grid()) {
        // TODO n < m ???
        matrix_type _Q = problem.input_matrix;
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

        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, _A, B, X);
        base::Trsm(elem::LEFT, elem::UPPER, elem::ADJOINT, elem::NON_UNIT,
            1.0, _R, X);
        base::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X);
    }

};

/**
 * Regression solver for L2 linear regrssion
 * on a dense distributed [VC/VR, STAR] that is not given explictly,
 * but rather as a "computed matrix". This implementation uses
 * an semi-normal equations based approach.
 *
 * NOTE: this solver keeps a reference to the input matrix (needs to be
 * kept in memory).
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType, elem::Distribution VD>
class regression_solver_t<
    regression_problem_t<
        base::computed_matrix_t< elem::DistMatrix<ValueType, VD, elem::STAR> >,
        linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    sne_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef base::computed_matrix_t< elem::DistMatrix<ValueType, VD, elem::STAR> >
    matrix_type;
    typedef elem::DistMatrix<ValueType, VD, elem::STAR> rhs_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    const matrix_type& _A;
    sol_type _R;

public:
    /**
     * Prepares the solver to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    regression_solver_t(const problem_type& problem) :
        _m(problem.m), _n(problem.n),_A(problem.input_matrix)  {
        // TODO n < m ???
        elem::DistMatrix<ValueType, VD, elem::STAR> _Q =
            problem.input_matrix.materialize();
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

        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, _A, B, X);
        base::Trsm(elem::LEFT, elem::UPPER, elem::ADJOINT, elem::NON_UNIT,
            1.0, _R, X);
        base::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X);
    }

};

} } /** namespace skylark::algorithms */

#endif // SKYLARK_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP
