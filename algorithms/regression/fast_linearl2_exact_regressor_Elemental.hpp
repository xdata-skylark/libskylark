#ifndef SKYlARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP
#define SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "regression_problem.hpp"

namespace skylark {
namespace algorithms {

template <typename ValueType, elem::Distribution VD,
          template <typename, typename> class TransformType>
class fast_exact_regressor_t<
    regression_problem_t<elem::DistMatrix<ValueType, VD, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    simplified_blendenpik_tag<TransformType> > {

public:

    typedef ValueType value_type;

    typedef elem::DistMatrix<ValueType, VD, elem::STAR> matrix_type;
    typedef elem::DistMatrix<ValueType, VD, elem::STAR> rhs_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> precond_type;

    const int _m;
    const int _n;
    const matrix_type &_A;
    precond_type _R;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     */
    fast_exact_regressor_t(const problem_type& problem, base::context_t& context) :
        _m(problem.m), _n(problem.n), _A(problem.input_matrix),
        _R(_n, _n, problem.input_matrix.Grid()) {
        // TODO n < m ???

        typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> sketch_type;

        int t = 4 * _n;    // TODO parameter.

        TransformType<matrix_type, sketch_type> S(_m, t, context);
        sketch_type SA(t, _n);
        S.apply(_A, SA, sketch::columnwise_tag());
        elem::qr::Explicit(SA.Matrix(), _R.Matrix());
    }

    int solve(const rhs_type& b, sol_type& x) {
        nla::tri_inverse_precond_t<sol_type, precond_type,
                                   elem::UPPER, elem::NON_UNIT> PR(_R);
        return LSQR(_A, b, x, nla::iter_params_t(), PR);
    }
};


} } /** namespace skylark::algorithms */

#endif // SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP
