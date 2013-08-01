#ifndef EXACT_REGRESSOR_ELEMENTAL_HPP
#define EXACT_REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "config.h"
#include "regression_problem.hpp"


namespace skylark {
namespace algorithms {

/**
 * Exact regressor (solves the problem exactly) for a dense local matrix.
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType>
class exact_regressor_t<l2_tag,
                        elem::Matrix<ValueType>,
                        elem::Matrix<ValueType>, 
                        qr_l2_solver_tag> {

    typedef elem::Matrix<ValueType> matrix_type;
    typedef elem::Matrix<ValueType> rhs_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _R;


public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    exact_regressor_t(const regression_problem_t<l2_tag, matrix_type> &problem) :
        _m(problem.m), _n(problem.n) {
        // TODO n < m
        _QR = problem.input_matrix;
        elem::QR(_QR);
        elem::LockedView(_R, _QR, 0, 0, _n, _n);
    }

    /**
     * Solves the regression problem given a single vector right-hand side.
     *
     * @param b Right-hand side.
     * @param x Output (overwritten).
     */
    void solve(const rhs_type &b, rhs_type &x) {
        // TODO error checking
        x = b;
        // TODO not sure the following like will work if b is a row vector
        elem::ApplyPackedReflectors(elem::LEFT, elem::LOWER,
            elem::VERTICAL, elem::FORWARD, 0, _QR, x);
        if (b.Width() == 1)
            x.ResizeTo(_n, 1);
        else
            x.ResizeTo(1, _n);
        elem::Trsv(elem::UPPER, elem::NORMAL, elem::NON_UNIT, _R, x);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve_mulitple(const rhs_type &B, rhs_type &X) {
        // TODO error checking
        X = B;
        elem::ApplyPackedReflectors(elem::LEFT, elem::LOWER,
            elem::VERTICAL, elem::FORWARD, 0, _QR, X);
        X.ResizeTo(_n, B.Width());
        elem::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X, true);
    }
};

/**
 * Exact regressor (solves the problem exactly) for a dense distributed matrix.
 *
 * A regressor accepts a right-hand side and output a solution
 * the the regression problem.
 *
 * The regression problem is fixed, so it is a parameter of the function
 * constructing the regressoion.
 */
template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RW>
class exact_regressor_t<l2_tag,
                        elem::DistMatrix<ValueType, CD, RW>,
                        elem::DistMatrix<ValueType, CD, RW>,
                        qr_l2_solver_tag> {

    typedef elem::DistMatrix<ValueType, CD, RW> matrix_type;
    typedef elem::DistMatrix<ValueType, CD, RW> rhs_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _R;


public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    exact_regressor_t(const regression_problem_t<l2_tag, matrix_type> &problem) :
        _m(problem.m), _n(problem.n),
        _QR(problem.input_matrix.Grid()), _R(problem.input_matrix.Grid()) {
        // TODO n < m
        _QR = problem.input_matrix;
        elem::QR(_QR);
        elem::LockedView(_R, _QR, 0, 0, _n, _n);
    }

    /**
     * Solves the regression problem given a single vector right-hand side.
     *
     * @param b Right-hand side.
     * @param x Output (overwritten).
     */
    void solve(const rhs_type &b, rhs_type &x) {
        // TODO error checking
        x = b;
        // TODO not sure the following like will work if b is a row vector
        elem::ApplyPackedReflectors(elem::LEFT, elem::LOWER,
            elem::VERTICAL, elem::FORWARD, 0, _QR, x);
        if (b.Width() == 1)
            x.ResizeTo(_n, 1);
        else
            x.ResizeTo(1, _n);
        elem::Trsv(elem::UPPER, elem::NORMAL, elem::NON_UNIT, _R, x);
    }

    /**
     * Solves the regression problem given a multiple right-hand sides.
     *
     * @param B Right-hand sides.
     * @param X Output (overwritten).
     */
    void solve_mulitple(const rhs_type &B, rhs_type &X) {
        // TODO error checking
        X = B;
        elem::ApplyPackedReflectors(elem::LEFT, elem::LOWER,
            elem::VERTICAL, elem::FORWARD, 0, _QR, X);
        X.ResizeTo(_n, B.Width());
        elem::Trsm(elem::LEFT, elem::UPPER, elem::NORMAL, elem::NON_UNIT,
            1.0, _R, X, true);
    }
};

} // namespace algorithms
} // namespace skylark

#endif // EXACT_REGRESSOR_ELEMENTAL_HPP
