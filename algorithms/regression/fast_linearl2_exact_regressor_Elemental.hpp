#ifndef SKYlARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP
#define SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP

#include <elemental.hpp>

#include "regression_problem.hpp"

namespace skylark {
namespace algorithms {

namespace internal {

template<typename SolType, typename SketchType, typename PrecondType>
void build_precond(SketchType& SA,
    PrecondType& R, nla::precond_t<SolType> *&P, qr_precond_tag) {
    elem::qr::Explicit(SA.Matrix(), R.Matrix()); // TODO
    P =
        new nla::tri_inverse_precond_t<SolType, PrecondType,
                                       elem::UPPER, elem::NON_UNIT>(R);
}

template<typename SolType, typename SketchType, typename PrecondType>
void build_precond(SketchType& SA,
    PrecondType& V, nla::precond_t<SolType> *&P, svd_precond_tag) {

    int n = SA.Width();
    PrecondType s(SA);
    s.Resize(n, 1);
    elem::SVD(SA.Matrix(), s.Matrix(), V.Matrix()); // TODO
    for(int i = 0; i < n; i++)
        s.Set(i, 0, 1 / s.Get(i, 0));
    base::DiagonalScale(elem::RIGHT, elem::NORMAL, s, V);
    P = 
        new nla::mat_precond_t<SolType, PrecondType>(V);
}

}  // namespace internal

/// Specialization for simplified blendenpik algorithm
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

        internal::build_precond(SA, _R, _precond_R, PrecondTag());
    }

    ~fast_exact_regressor_t() {
        delete _precond_R;
    }

    int solve(const rhs_type& b, sol_type& x) {
        return LSQR(_A, b, x, nla::iter_params_t(), *_precond_R);
    }
};

/// Specialization for LSRN algorithm.
template <typename ValueType, elem::Distribution VD,
          typename PrecondTag>
class fast_exact_regressor_t<
    regression_problem_t<elem::DistMatrix<ValueType, VD, elem::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    elem::DistMatrix<ValueType, VD, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    lsrn_tag<PrecondTag> > {

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
    bool _use_lsqr;
    double _sigma_U, _sigma_L;
    precond_type _R;
    nla::precond_t<sol_type> *_precond_R;
    nla::iter_params_t _params;

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
        double epsilon = 1e-14;  // TODO parameter
        double delta = 1e-6; // TODO parameter

        sketch::JLT_t<matrix_type, sketch_type> S(_m, t, context);
        sketch_type SA(t, _n);
        S.apply(_A, SA, sketch::columnwise_tag());
        internal::build_precond(SA, _R, _precond_R, PrecondTag());

        // Select alpha so that probability of failure is delta.
        // If alpha is too big, we need to use LSQR (although ill-conditioning
        // might be very severe so to prevent convergence).
        double alpha = std::sqrt(2 * std::log(2.0 / delta) / t);
        if (alpha >= (1 - std::sqrt(_n / t)))
            _use_lsqr = true;
        else {
            _use_lsqr = false;
            _sigma_U = std::sqrt(t) / ((1 - alpha) * std::sqrt(t)
                - std::sqrt(_n));
            _sigma_L = std::sqrt(t) / ((1 + alpha) * std::sqrt(t)
                + std::sqrt(_n));
        }
    }

    ~fast_exact_regressor_t() {
        delete _precond_R;
    }

    int solve(const rhs_type& b, sol_type& x) {
        int ret;
        if (_use_lsqr)
            ret = LSQR(_A, b, x, _params, *_precond_R);
        else {
            ChebyshevLS(_A, b, x,  _sigma_L, _sigma_U,
                _params, *_precond_R);
            ret = -6; // TODO! - check!
        }
        return ret; // TODO!
    }
};



} } /** namespace skylark::algorithms */

#endif // SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_ELEMENTAL_HPP
