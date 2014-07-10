#ifndef SKYLARK_CG_HPP
#define SKYLARK_CG_HPP

#include "../../base/base.hpp"
#include "../../utility/elem_extender.hpp"
#include "../../utility/typer.hpp"
#include "../../utility/external/print.hpp"
#include "internal.hpp"
#include "precond.hpp"

namespace skylark { namespace algorithms {

// We can have a version that is indpendent of Elemental. But that will
// be tedious (convert between [STAR,STAR] and vector<T>, and really
// elemental is a very fudmanetal to Skylark.
#if SKYLARK_HAVE_ELEMENTAL

/**
 * CG method.
 *
 * X should be allocated, and we use it as initial value.
 */
template<typename MatrixType, typename RhsType, typename SolType>
int CG(const MatrixType& A, const RhsType& B, SolType& X,
    krylov_iter_params_t params = krylov_iter_params_t(),
    const precond_t<SolType>& M = id_precond_t<SolType>()) {

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef RhsType rhs_type;
    typedef SolType sol_type;

    typedef utility::print_t<rhs_type> rhs_print_t;
    typedef utility::print_t<sol_type> sol_print_t;

    typedef utility::elem_extender_t<
        typename internal::scalar_cont_typer_t<rhs_type>::type >
        scalar_cont_type;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    /** Throughout, we will use n, k to denote the problem dimensions */
    index_t n = base::Height(A);
    index_t k = base::Width(B);

    /** Set the parameter values accordingly */
    const value_t eps = 32*std::numeric_limits<value_t>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    sol_type P(X);
    rhs_type R(B), Q(B);
    bool isprecond = !(M.is_id() && std::is_same<sol_type, rhs_type>::value);
    sol_type &Z =  !isprecond ? R : *(new sol_type(X));

    base::Gemm(elem::NORMAL, elem::NORMAL, -1.0, A, X, 1.0, R);

    scalar_cont_type
        nrmb(internal::scalar_cont_typer_t<rhs_type>::build_compatible(k, 1, B));
    base::ColumnNrm2(B, nrmb);
    scalar_cont_type ressqr(nrmb), rho(nrmb), rho0(nrmb), rhotmp(nrmb),
        alpha(nrmb), malpha(nrmb), beta(nrmb);
    base::ColumnDot(R, R, ressqr);

    for (index_t itn=0; itn<params.iter_lim; ++itn) {
        if (isprecond) {
            Z = R;
            M.apply(Z);
            base::ColumnDot(R, Z, rho);
        } else
            rho = ressqr;

        if (itn == 0)
            elem::MakeZeros(beta);
        else
            for(index_t i = 0; i < k; i++)
                beta[i] = rho[i] / rho0[i];

        base::DiagonalScale(elem::RIGHT, elem::NORMAL, beta, P);
        base::Axpy(1.0, Z, P);

        base::Gemm(elem::NORMAL, elem::NORMAL, 1.0, A, P, Q);

        base::ColumnDot(P, Q, rhotmp);
        for(index_t i = 0; i < k; i++) {
            alpha[i] = rho[i] / rhotmp[i];
            malpha[i] = -alpha[i];
        }

        base::Axpy(alpha, P, X);
        base::Axpy(malpha, Q, R);

        rho0 = rho;

        base::ColumnDot(R, R, ressqr);

        int convg = 0;
        for(index_t i = 0; i < k; i++)
            if (sqrt(ressqr[i]) < (params.tolerance*nrmb[i]))
                convg++;
        if(convg == k) {
            if (log_lev1)
                params.log_stream << "CG: Convergence!" << std::endl;
            return -1;
        }
    }

    if (isprecond)
        delete &Z;

   if (log_lev1)
        params.log_stream << "CG: No convergence within iteration limit."
                          << std::endl;

    return -6;
}

#endif

} } /** namespace skylark::algorithms */

#endif // SKYLARK_CG_HPP
