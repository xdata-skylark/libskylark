#ifndef SKYLARK_FLEXIBLE_CG_HPP
#define SKYLARK_FLEXIBLE_CG_HPP

#include "../../base/base.hpp"
#include "../../utility/elem_extender.hpp"
#include "../../utility/typer.hpp"
#include "../../utility/external/print.hpp"
#include "internal.hpp"
#include "precond.hpp"

namespace skylark { namespace algorithms {

/**
 * FlexibleCG method.
 *
 * The method is normally applied to SPD matrix A. Only a triangular
 * part of the matrix is accessed.
 *
 * X should be allocated, and we use it as initial value.
 */
template<typename MatrixType, typename RhsType, typename SolType>
int FlexibleCG(El::UpperOrLower uplo, const MatrixType& A, const RhsType& B, SolType& X,
    krylov_iter_params_t params = krylov_iter_params_t(),
    const outplace_precond_t<RhsType, SolType>& M =
    outplace_id_precond_t<RhsType, SolType>()) {

    int ret;

    typedef typename utility::typer_t<MatrixType>::value_type value_type;
    typedef typename utility::typer_t<MatrixType>::index_type index_type;

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
    index_type n = base::Height(A);
    index_type k = base::Width(B);

    index_type maxit = params.iter_lim;

    /** Set the parameter values accordingly */
    const value_type eps = 32*std::numeric_limits<value_type>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    rhs_type R(B), l(B);
    sol_type *D = new sol_type[maxit];
    rhs_type *L = new rhs_type[maxit];
    for(index_type i = 0; i < maxit; i++) {
        D[i] = X;
        L[i] = B;
    }

    scalar_cont_type
        nrmb(internal::scalar_cont_typer_t<rhs_type>::build_compatible(k, 1, B));
    scalar_cont_type ressqr(nrmb), beta(nrmb),  alpha(nrmb),
        malpha(nrmb), gamma(nrmb), mgamma(nrmb);

    base::Symm(El::LEFT, uplo, value_type(-1.0), A, X, value_type(1.0), R);
    base::ColumnNrm2(B, nrmb);
    double total_nrmb = 0.0;
    for(index_type i = 0; i < k; i++)
        total_nrmb += nrmb[i] * nrmb[i];
    total_nrmb = sqrt(total_nrmb);
    base::ColumnDot(R, R, ressqr);

    for (index_type itn=0; itn<params.iter_lim; ++itn) {
        sol_type &d = D[itn];
        rhs_type &l = L[itn];

        M.apply(R, d);

        // TODO. The following is Modified Gram-Schmidt. In terms of
        // synchronization points, this is really bad, so we might want to
        // replace with a CGS procedure instead.
        for(index_type i = 0; i < itn; i++) {
            base::ColumnDot(L[i], d, gamma);
            for(index_type r = 0; r < k; r++)
                mgamma[r] = -gamma[r];
            base::Axpy(mgamma, D[i], d);
        }

	base::Symm(El::LEFT, uplo, value_type(1.0), A, d, l);

        base::ColumnDot(d, l, beta);
        base::ColumnDot(d, R, alpha);

        for(index_type i = 0; i < k; i++) {
            beta[i] = 1 / sqrt(beta[i]);
            alpha[i] *= beta[i];
            malpha[i] = -alpha[i];
        }

        El::DiagonalScale(El::RIGHT, El::NORMAL, beta, d);
        El::DiagonalScale(El::RIGHT, El::NORMAL, beta, l);

        base::Axpy(alpha, d, X);
        base::Axpy(malpha, l, R);

        base::ColumnDot(R, R, ressqr);

        int convg = 0;
        for(index_type i = 0; i < k; i++)
            if (sqrt(ressqr[i]) < (params.tolerance*nrmb[i]))
                convg++;

        if (log_lev2 && (itn % params.res_print == 0 || convg == k)) {
            double total_ressqr = 0.0;
            for(index_type i = 0; i < k; i++)
                total_ressqr += ressqr[i];
            double relres = sqrt(total_ressqr) / total_nrmb;
            params.log_stream << params.prefix
                              << "FlexibleCG: Iteration " << itn
                              << ", Relres = "
                              << boost::format("%.2e") % relres
                              << ", " << convg << " rhs converged" << std::endl;
        }

        if(convg == k) {
            if (log_lev1)
                params.log_stream << params.prefix
                                  << "FlexibleCG: Convergence!" << std::endl;
            ret = -1;
            goto cleanup;
        }
    }


   if (log_lev1)
        params.log_stream << params.prefix
                          << "FlexibleCG: No convergence within iteration limit."
                          << std::endl;
   ret = -6;

 cleanup:
   delete []D;
   delete []L;

   return ret;
}

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
    const outplace_precond_t<RhsType, SolType>& M =
    outplace_id_precond_t<RhsType, SolType>()) {

    int ret;

    typedef typename utility::typer_t<MatrixType>::value_type value_type;
    typedef typename utility::typer_t<MatrixType>::index_type index_type;

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
    index_type n = base::Height(A);
    index_type k = base::Width(B);

    index_type maxit = params.iter_lim;

    /** Set the parameter values accordingly */
    const value_type eps = 32*std::numeric_limits<value_type>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    rhs_type R(B), l(B);
    sol_type *D = new sol_type[maxit];
    rhs_type *L = new rhs_type[maxit];
    for(index_type i = 0; i < maxit; i++) {
        D[i] = X;
        L[i] = B;
    }

    scalar_cont_type
        nrmb(internal::scalar_cont_typer_t<rhs_type>::build_compatible(k, 1, B));
    scalar_cont_type ressqr(nrmb), beta(nrmb),  alpha(nrmb),
        malpha(nrmb), gamma(nrmb), mgamma(nrmb);

    base::Gemm(El::ADJOINT, El::NORMAL, value_type(-1.0), A, X, value_type(1.0), R);
    base::ColumnNrm2(B, nrmb);
    double total_nrmb = 0.0;
    for(index_type i = 0; i < k; i++)
        total_nrmb += nrmb[i] * nrmb[i];
    total_nrmb = sqrt(total_nrmb);
    base::ColumnDot(R, R, ressqr);

    for (index_type itn=0; itn<params.iter_lim; ++itn) {
        sol_type &d = D[itn];
        rhs_type &l = L[itn];

        M.apply(R, d);

        // TODO. The following is Modified Gram-Schmidt. In terms of
        // synchronization points, this is really bad, so we might want to
        // replace with a CGS procedure instead.
        for(index_type i = 0; i < itn; i++) {
            base::ColumnDot(L[i], d, gamma);
            for(index_type r = 0; r < k; r++)
                mgamma[r] = -gamma[r];
            base::Axpy(mgamma, D[i], d);
        }

	base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), A, d, l);

        base::ColumnDot(d, l, beta);
        base::ColumnDot(d, R, alpha);

        for(index_type i = 0; i < k; i++) {
            beta[i] = 1 / sqrt(beta[i]);
            alpha[i] *= beta[i];
            malpha[i] = -alpha[i];
        }

        El::DiagonalScale(El::RIGHT, El::NORMAL, beta, d);
        El::DiagonalScale(El::RIGHT, El::NORMAL, beta, l);

        base::Axpy(alpha, d, X);
        base::Axpy(malpha, l, R);

        base::ColumnDot(R, R, ressqr);

        int convg = 0;
        for(index_type i = 0; i < k; i++)
            if (sqrt(ressqr[i]) < (params.tolerance*nrmb[i]))
                convg++;

        if (log_lev2 && (itn % params.res_print == 0 || convg == k)) {
            double total_ressqr = 0.0;
            for(index_type i = 0; i < k; i++)
                total_ressqr += ressqr[i];
            double relres = sqrt(total_ressqr) / total_nrmb;
            params.log_stream << params.prefix
                              << "FlexibleCG: Iteration " << itn
                              << ", Relres = "
                              << boost::format("%.2e") % relres
                              << ", " << convg << " rhs converged" << std::endl;
        }

        if(convg == k) {
            if (log_lev1)
                params.log_stream << params.prefix
                                  << "FlexibleCG: Convergence!" << std::endl;
            ret = -1;
            goto cleanup;
        }
    }


   if (log_lev1)
        params.log_stream << params.prefix
                          << "FlexibleCG: No convergence within iteration limit."
                          << std::endl;
   ret = -6;

 cleanup:
   delete []D;
   delete []L;

   return ret;
}


} } /** namespace skylark::algorithms */

#endif // SKYLARK_FLEXIBLE_CG_HPP
