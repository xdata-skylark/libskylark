#ifndef SKYlARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP
#define SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "regression_problem.hpp"

namespace skylark {
namespace algorithms {

template <typename ValueType, elem::Distribution VD,
          template <typename, typename> class TransformType,
          typename PrecondTag>
class fast_exact_regressor_t<
    regression_problem_t<elem::DistMatrix<ValueType, VD, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    simplified_blendenpik_tag<TransformType, PrecondTag> > {

public:

    typedef ValueType value_type;

    typedef elem::DistMatrix<ValueType, VD, elem::STAR> matrix_type;
    typedef elem::DistMatrix<ValueType, VD, elem::STAR> rhs_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef elem::DistMatrix<ValueType, elem::STAR, elem::STAR> precond_type;
    typedef precond_type sketch_type; 
    // The assumption is that the sketch is not much bigger than the
    // preconditioner, so we should use the same matrix distribution.

    const int _m;
    const int _n;
    const matrix_type &_A;
    precond_type _R;
    nla::precond_t<sol_type> *_precond_R;

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

        int t = 4 * _n;    // TODO parameter.

        TransformType<matrix_type, sketch_type> S(_m, t, context);
        sketch_type SA(t, _n);
        S.apply(_A, SA, sketch::columnwise_tag());

        _precond_R = build_precond(SA, _R, PrecondTag());
    }

    ~fast_exact_regressor_t() {
        delete _precond_R;
    }

    int solve(const rhs_type& b, sol_type& x) {
        return LSQR(_A, b, x, nla::iter_params_t(), *_precond_R);
    }

private:
    static nla::precond_t<sol_type> *build_precond(sketch_type& SA,
        precond_type& R, qr_precond_tag) {
        elem::qr::Explicit(SA.Matrix(), R.Matrix()); // TODO
        return
            new nla::tri_inverse_precond_t<sol_type, precond_type,
                                             elem::UPPER, elem::NON_UNIT>(R);
    }

    static nla::precond_t<sol_type> *build_precond(sketch_type& SA,
        precond_type& V, svd_precond_tag) {

        int n = SA.Width();
        precond_type s(n, 1, SA.Grid());
        elem::SVD(SA.Matrix(), s.Matrix(), V.Matrix()); // TODO
        for(int i = 0; i < n; i++)
            s.Set(i, 0, 1 / s.Get(i, 0));
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, s, V);
        return
            new nla::mat_precond_t<sol_type, precond_type>(V);
    }
};


} } /** namespace skylark::algorithms */

#endif // SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP
