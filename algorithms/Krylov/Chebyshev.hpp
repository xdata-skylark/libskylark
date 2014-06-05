#ifndef SKYLARK_CHEBYSHEV_HPP
#define SKYLARK_CHEBYSHEV_HPP

#include "../../base/base.hpp"
#include "../../utility/elem_extender.hpp"
#include "../../utility/typer.hpp"
#include "../../utility/external/print.hpp"
#include "precond.hpp"

namespace skylark { namespace nla {

// We can have a version that is indpendent of Elemental. But that will
// be tedious (convert between [STAR,STAR] and vector<T>, and really
// elemental is a very fudmanetal to Skylark.
#if SKYLARK_HAVE_ELEMENTAL

/**
 * Chebyshev Semi-Iterative method.
 *
 * X should be allocated, but we zero it on start. (not set as X_0).
 */
template<typename MatrixType, typename RhsType, typename SolType>
void ChebyshevLS(const MatrixType& A, const RhsType& B, SolType& X,
    double sigma_L, double sigma_U,
    iter_params_t params = iter_params_t(),
    const precond_t<SolType>& P = id_precond_t<SolType>()) {

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef RhsType rhs_type;        // Also serves as "long" vector type.
    typedef SolType sol_type;        // Also serves as "short" vector type.

    typedef utility::print_t<rhs_type> rhs_print_t;
    typedef utility::print_t<sol_type> sol_print_t;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    /** Throughout, we will use m, n, k to denote the problem dimensions */
    index_t m = base::Height(A);
    index_t n = base::Width(A);
    index_t k = base::Width(B);

    /** Set the parameter values accordingly */
    const value_t eps = 32*std::numeric_limits<value_t>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    double its = (std::log(params.tolerance) - std::log(2))
        / std::log((sigma_U - sigma_L)/(sigma_U + sigma_L)) + 1;

    double d = (sigma_U * sigma_U + sigma_L * sigma_L) / 2;
    double c = (sigma_U * sigma_U - sigma_L * sigma_L) / 2;

    base::Zero(X);
    rhs_type R(B);
    sol_type V(X), AR(X);

    double alpha = 0.0, beta = 0.0;
    for(int i = 0; i < its; i++) {
        switch(i) {
        case 0:
            beta = 0.0;
            alpha = 1.0 / d;
            break;

        case 1:
            beta = (c * c) / (d * d * 2);
            alpha = 1 / (d - c * c / (2 * d));
            break;

        default:
            beta = alpha * alpha * c * c / 4.0;
            alpha = 1 / (d - alpha * c * c / 4.0);
            break;
        }

        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, A, R, AR);
        P.apply_adjoint(AR);
        base::Scal(beta, V);
        base::Axpy(1.0, AR, V);
        P.apply(V);
        base::Axpy(alpha, V, X);
        base::Gemm(elem::NORMAL, elem::NORMAL, -alpha, A, V, 1.0, R);
    }
}

#endif

} } /** namespace skylark::nla */

#endif // SKYLARK_CHEBYSHEV_HPP
