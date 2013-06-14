#ifndef REGRESSOR_ELEMENTAL_HPP
#define REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "config.h"
#include "regression_problem.hpp"
#include "sketch/sketch.hpp"


namespace skylark {
namespace algorithms {

/**
 * Exact l2 regressor using Elemental.
 */
template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RW>
class exact_regressor_t<l2_tag,
    elem::DistMatrix<ValueType, CD, RW>,
    elem::DistMatrix<ValueType, CD, RW> > {

    typedef elem::DistMatrix<ValueType, CD, RW> matrix_type;
    typedef elem::DistMatrix<ValueType, CD, RW> rhs_type;

private:
    const int _m;
    const int _n;
    matrix_type _QR;
    matrix_type _R;


public:
    exact_regressor_t(const regression_problem_t<l2_tag, matrix_type> &problem) :
        _m(problem.m), _n(problem.n),
        _QR(problem.input_matrix.Grid()), _R(problem.input_matrix.Grid()) {
        // TODO n < m
        _QR = problem.input_matrix;
        elem::QR(_QR);
        elem::LockedView(_R, _QR, 0, 0, _n, _n);
    }

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
