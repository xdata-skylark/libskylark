#ifndef SKYLARK_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP
#define SKYLARK_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP

#include <El.hpp>
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
    regression_problem_t<El::Matrix<ValueType>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::Matrix<ValueType>,
    El::Matrix<ValueType>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::Matrix<ValueType> matrix_type;
    typedef El::Matrix<ValueType> rhs_type;
    typedef El::Matrix<ValueType> sol_type;

    typedef regression_problem_t<
        El::Matrix<ValueType>, linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _t, _d;
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
        El::QR(_QR, _t, _d);
        El::LockedView(_R, _QR, 0, 0, _n, _n);
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
        El::qr::ApplyQ(El::LEFT, El::ADJOINT, _QR, _t, _d, X);
        X.Resize(_n, B.Width());
        El::Trsm(El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
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
    regression_problem_t<El::DistMatrix<ValueType, El::STAR, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

    typedef regression_problem_t<
        El::DistMatrix<ValueType, El::STAR, El::STAR>,
        linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _t, _d;
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
        // TODO in 0.85, still use Matrix()?
        El::QR(_QR.Matrix(), _t.Matrix(), _d.Matrix());
        El::LockedView(_R.Matrix(), _QR.Matrix(), 0, 0, _n, _n);
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
        El::qr::ApplyQ(El::LEFT, El::ADJOINT,
            _QR.LockedMatrix(), _t.LockedMatrix(), _d.LockedMatrix(), X.Matrix());
        X.Resize(_n, B.Width());
        El::Trsm(El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
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
    regression_problem_t<El::DistMatrix<ValueType>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType>,
    El::DistMatrix<ValueType>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType> matrix_type;
    typedef El::DistMatrix<ValueType> rhs_type;
    typedef El::DistMatrix<ValueType> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _R;
    El::DistMatrix<ValueType, El::MD, El::STAR> _t, _d; // TODO: still use MD,STAR?

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
        El::QR(_QR, _t, _d);
        El::LockedView(_R, _QR, 0, 0, _n, _n);
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
        El::qr::ApplyQ(El::LEFT, El::ADJOINT, _QR, _t, _d, X);
        X.Resize(_n, B.Width());
        El::Trsm(El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
            value_type(1.0), _R, X);
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
template <typename ValueType, El::Distribution VD>
class regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, VD, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    qr_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, VD, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

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
        El::qr::ExplicitTS(_Q, _R);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve (const rhs_type& B, sol_type& X) const {
        // TODO error checking

        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), _Q, B, X);
        base::Trsm(El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
            value_type(1.0), _R, X);
    }

};

/**
 * Regression solver for L2 linear regrssion
 * on a dense sequential matrices. This implementation uses
 * an SVD-based approach.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class regression_solver_t<
    regression_problem_t<El::Matrix<ValueType>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::Matrix<ValueType>,
    El::Matrix<ValueType>,
    svd_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::Matrix<ValueType> matrix_type;
    typedef El::Matrix<ValueType> rhs_type;
    typedef El::Matrix<ValueType> sol_type;

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
        _U(), _S(), _V() {
        // TODO n < m ???
        El::SVD(problem.input_matrix, _U, _S, _V);
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
        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), _U, B, UB);
        El::DiagonalScale(El::LEFT, El::NORMAL, _S, UB);
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), _V, UB, X);
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
template <typename ValueType, El::Distribution VD>
class regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, VD, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    svd_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, VD, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

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
        El::SVD(problem.input_matrix, _U, _S, _V);
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
        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), _U, B, UB);
        El::DiagonalScale(El::LEFT, El::NORMAL, _S, UB);
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), _V, UB, X);
    }

};

/**
 * Regression solver for L2 linear regrssion
 * on a dense distributed all-same matrices. This implementation uses
 * an SVD-based approach.
 *
 * A regression solver accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType, El::Distribution U, El::Distribution V>
class regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, U, V>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, U, V>,
    El::DistMatrix<ValueType, U, V>,
    svd_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, U, V> matrix_type;
    typedef El::DistMatrix<ValueType, U, V> rhs_type;
    typedef El::DistMatrix<ValueType, U, V> sol_type;

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
        El::SVD(problem.input_matrix, _U, _S, _V);
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
        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), _U, B, UB);
        El::DiagonalScale(El::LEFT, El::NORMAL, _S, UB);
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), _V, UB, X);
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
template <typename ValueType, El::Distribution VD>
class regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, VD, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    sne_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, VD, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

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
        El::qr::ExplicitTS(_Q, _R);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve (const rhs_type& B, sol_type& X) const {
        // TODO error checking

        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), _A, B, X);
        base::Trsm(El::LEFT, El::UPPER, El::ADJOINT, El::NON_UNIT,
            value_type(1.0), _R, X);
        base::Trsm(El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
            value_type(1.0), _R, X);
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
template <typename ValueType, El::Distribution VD>
class regression_solver_t<
    regression_problem_t<
        base::computed_matrix_t< El::DistMatrix<ValueType, VD, El::STAR> >,
        linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    sne_l2_solver_tag> {

public:

    typedef ValueType value_type;

    typedef base::computed_matrix_t< El::DistMatrix<ValueType, VD, El::STAR> >
    matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

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
        El::DistMatrix<ValueType, VD, El::STAR> _Q =
            problem.input_matrix.materialize();
        El::qr::ExplicitTS(_Q, _R);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve (const rhs_type& B, sol_type& X) const {
        // TODO error checking

        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), _A, B, X);
        base::Trsm(El::LEFT, El::UPPER, El::ADJOINT, El::NON_UNIT,
            value_type(1.0), _R, X);
        base::Trsm(El::LEFT, El::UPPER, El::NORMAL, El::NON_UNIT,
            value_type(1.0), _R, X);
    }

};

} } /** namespace skylark::algorithms */

#endif // SKYLARK_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP
