#ifndef SKYLARK_FLEXIBLE_CG_HPP
#define SKYLARK_FLEXIBLE_CG_HPP

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
 * FlexibleCG method.
 *
 * The method is normally applied to SPD matrix A. However, you can
 * attempt to apply it even for a non-symmetric matrix, but be aware
 * that the code will operate actually on A^T in that case.
 *
 * X should be allocated, and we use it as initial value.
 */
template<typename MatrixType, typename RhsType, typename SolType>
int FlexibleCG(const MatrixType& A, const RhsType& B, SolType& X,
    krylov_iter_params_t params = krylov_iter_params_t(),
    const precond_t<SolType>& M = id_precond_t<SolType>()) {

    int ret;

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

    index_t maxit = params.iter_lim;

    /** Set the parameter values accordingly */
    const value_t eps = 32*std::numeric_limits<value_t>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    rhs_type R(B), l(B);
    sol_type *D = new sol_type[maxit];
    rhs_type *L = new rhs_type[maxit];
    for(index_t i = 0; i < maxit; i++) {
        D[i] = X;
        L[i] = B;
    }

    scalar_cont_type
        nrmb(internal::scalar_cont_typer_t<rhs_type>::build_compatible(k, 1, B));
    scalar_cont_type ressqr(nrmb), beta(nrmb),  alpha(nrmb),
        malpha(nrmb), gamma(nrmb), mgamma(nrmb);

    base::Gemm(elem::ADJOINT, elem::NORMAL, -1.0, A, X, 1.0, R);
    base::ColumnNrm2(B, nrmb);
    base::ColumnDot(R, R, ressqr);

    for (index_t itn=0; itn<params.iter_lim; ++itn) {
        sol_type &d = D[itn];
        rhs_type &l = L[itn];

        elem::Copy(R, d);
        M.apply(d);  // TODO it might be better to pass two parameters...

        // TODO. The following is Modified Gram-Schmidt. In terms of
        // sycnrhonization points, this is really bad, so we might want to
        // replace with a CGS procedure instead.
        for(index_t i = 0; i < itn; i++) {
            base::ColumnDot(L[i], d, gamma);
            for(index_t r = 0; r < k; r++)
                mgamma[r] = -gamma[r];
            base::Axpy(mgamma, D[i], d);
        }

        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, A, d, l);

        base::ColumnDot(d, l, beta);
        base::ColumnDot(d, R, alpha);

        for(index_t i = 0; i < k; i++) {
            beta[i] = 1 / sqrt(beta[i]);
            alpha[i] *= beta[i];
            malpha[i] = -alpha[i];
        }

        base::DiagonalScale(elem::RIGHT, elem::NORMAL, beta, d);
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, beta, l);

        base::Axpy(alpha, d, X);
        base::Axpy(malpha, l, R);

        base::ColumnDot(R, R, ressqr);

        int convg = 0;
        for(index_t i = 0; i < k; i++)
            if (sqrt(ressqr[i]) < (params.tolerance*nrmb[i]))
                convg++;
        if(convg == k) {
            if (log_lev1)
                params.log_stream << "FlexibleCG: Convergence!" << std::endl;
            ret = -1;
            goto cleanup;
        }
    }


   if (log_lev1)
        params.log_stream << "FelxibleCG: No convergence within iteration limit."
                          << std::endl;
   ret = -6;

 cleanup:
   delete []D;
   delete []L;

   return ret;
}

#endif

} } /** namespace skylark::algorithms */

#endif // SKYLARK_FLEXIBLE_CG_HPP
