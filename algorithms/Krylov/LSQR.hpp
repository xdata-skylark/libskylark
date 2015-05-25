#ifndef SKYLARK_LSQR_HPP
#define SKYLARK_LSQR_HPP

#include "../../base/base.hpp"
#include "../../utility/elem_extender.hpp"
#include "../../utility/typer.hpp"
#include "../../utility/external/print.hpp"
#include "internal.hpp"
#include "precond.hpp"

namespace skylark {
namespace algorithms {

/**
 * LSQR method.
 *
 * X should be allocated, but we zero it on start. (not set as X_0).
 */
template<typename MatrixType, typename RhsType, typename SolType>
int LSQR(const MatrixType& A, const RhsType& B, SolType& X,
    krylov_iter_params_t params = krylov_iter_params_t(),
    const inplace_precond_t<SolType>& R = inplace_id_precond_t<SolType>()) {

    typedef typename utility::typer_t<MatrixType>::value_type value_type;
    typedef typename utility::typer_t<MatrixType>::index_type index_type;

    typedef MatrixType matrix_type;
    typedef RhsType rhs_type;        // Also serves as "long" vector type.
    typedef SolType sol_type;        // Also serves as "short" vector type.

    typedef utility::print_t<rhs_type> rhs_print_t;
    typedef utility::print_t<sol_type> sol_print_t;

    typedef utility::elem_extender_t<
        typename internal::scalar_cont_typer_t<rhs_type>::type >
        scalar_cont_type;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    /** Throughout, we will use m, n, k to denote the problem dimensions */
    index_type m = base::Height(A);
    index_type n = base::Width(A);
    index_type k = base::Width(B);

    /** Set the parameter values accordingly */
    const value_type eps = 32*std::numeric_limits<value_type>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    /** Initialize everything */
    // We set the grid and rank for beta, and all other scalar containers
    // just copy from him to get that to be set right (not for the values).
    rhs_type U(B);
    scalar_cont_type
        beta(internal::scalar_cont_typer_t<rhs_type>::build_compatible(k, 1, U));
    scalar_cont_type i_beta(beta);
    base::ColumnNrm2(U, beta);
    for (index_type i=0; i<k; ++i)
        i_beta[i] = 1 / beta[i];
    El::DiagonalScale(El::RIGHT, El::NORMAL, i_beta, U);
    rhs_print_t::apply(U, "U Init", params.am_i_printing, params.debug_level);

    sol_type V(X);     // No need to really copy, just want sizes&comm correct.
    base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), A, U, V);
    R.apply_adjoint(V);
    scalar_cont_type alpha(beta), i_alpha(beta);
    base::ColumnNrm2(V, alpha);
    for (index_type i=0; i<k; ++i)
        i_alpha[i] = 1 / alpha[i];
    El::DiagonalScale(El::RIGHT, El::NORMAL, i_alpha, V);
    sol_type Z(V);
    R.apply(Z);
    sol_print_t::apply(V, "V Init", params.am_i_printing, params.debug_level);

    /* Create W=Z and X=0 */
    El::Zero(X);
    sol_type W(Z);
    scalar_cont_type phibar(beta), rhobar(alpha), nrm_r(beta);
        // /!\ Actually copied for init
    scalar_cont_type nrm_a(beta), cnd_a(beta), sq_d(beta), nrm_ar_0(beta);
    El::Zero(nrm_a); El::Zero(cnd_a); El::Zero(sq_d);
    El::Hadamard(alpha, beta, nrm_ar_0);

    /** Return from here */
    for (index_type i=0; i<k; ++i)
        if (nrm_ar_0[i]==0)
            return 0;

    scalar_cont_type nrm_x(beta), sq_x(beta), z(beta), cs2(beta), sn2(beta);
    El::Zero(nrm_x); El::Zero(sq_x); El::Zero(z); El::Zero(sn2);
    for (index_type i=0; i<k; ++i)
        cs2[i] = -1.0;

    int max_n_stag = 3;
    std::vector<int> stag(k, 0);

    /* Reset the iteration limit if none was specified */
    if (0>params.iter_lim) params.iter_lim = std::max(20, 2*std::min(m,n));

    /* More varaibles */
    sol_type AU(X);
    scalar_cont_type minus_beta(beta), rho(beta);
    scalar_cont_type cs(beta), sn(beta), theta(beta), phi(beta);
    scalar_cont_type phi_by_rho(beta), minus_theta_by_rho(beta), nrm_ar(beta);
    scalar_cont_type nrm_w(beta), sq_w(beta), gamma(beta);
    scalar_cont_type delta(beta), gambar(beta), rhs(beta), zbar(beta);

    /** Main iteration loop */
    for (index_type itn=0; itn<params.iter_lim; ++itn) {

        /** 1. Update u and beta */
        El::Scale(value_type(-1.0), alpha);
        // (Can safely overwrite based on subseq ops.)
        El::DiagonalScale(El::RIGHT, El::NORMAL, alpha, U);
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), A, Z,
            value_type(1.0), U);
        base::ColumnNrm2(U, beta);
        for (index_type i=0; i<k; ++i)
            i_beta[i] = 1 / beta[i];
        El::DiagonalScale(El::RIGHT, El::NORMAL, i_beta, U);

        /** 2. Estimate norm of A */
        for (index_type i=0; i<k; ++i) {
            double a = nrm_a[i], b = alpha[i], c = beta[i];
            nrm_a[i] = sqrt(a*a + b*b + c*c);
        }

        /** 3. Update v */
        for (index_type i=0; i<k; ++i)
            minus_beta[i] = -beta[i];
        El::DiagonalScale(El::RIGHT, El::NORMAL, minus_beta, V);
        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), A, U, AU);
        R.apply_adjoint(AU);
        base::Axpy(value_type(1.0), AU, V);
        base::ColumnNrm2(V, alpha);
        for (index_type i=0; i<k; ++i)
            i_alpha[i] = 1 / alpha[i];
        El::DiagonalScale(El::RIGHT, El::NORMAL, i_alpha, V);
        Z = V; R.apply(Z);

       /** 4. Define some variables */
        for (index_type i=0; i<k; ++i) {
            rho[i] = sqrt((rhobar[i]*rhobar[i]) + (beta[i]*beta[i]));
            cs[i] = rhobar[i]/rho[i];
            sn[i] =  beta[i]/rho[i];
            theta[i] = sn[i]*alpha[i];
            rhobar[i] = -cs[i]*alpha[i];
            phi[i] = cs[i]*phibar[i];
            phibar[i] =  sn[i]*phibar[i];
        }

        /** 5. Update X and W */
        for (index_type i=0; i<k; ++i)
            phi_by_rho[i] = phi[i]/rho[i];
        base::Axpy(phi_by_rho, W, X);
        sol_print_t::apply(X, "X", params.am_i_printing, params.debug_level);

        for (index_type i=0; i<k; ++i)
            minus_theta_by_rho[i] = -theta[i]/rho[i];
        El::DiagonalScale(El::RIGHT, El::NORMAL, minus_theta_by_rho, W);
        base::Axpy(value_type(1.0), Z, W);
        sol_print_t::apply(W, "W", params.am_i_printing, params.debug_level);

        /** 6. Estimate norm(r) */
        nrm_r = phibar;

        /** 7. estimate of norm(A'*r) */
        index_type cond_s1 = 0, cond_s2 = 0;
        for (index_type i=0; i<k; ++i) {
            nrm_ar[i] = std::abs(phibar[i]*alpha[i]*cs[i]);

            if (log_lev2)
                params.log_stream << params.prefix
                                  << "LSQR: Iteration " << i << "/" << itn
                                  << ": " << nrm_ar[i]
                                  << std::endl;


            if (nrm_ar[i]<(params.tolerance*nrm_ar_0[i]))
                cond_s1++;
            if (nrm_ar[i]<(eps*nrm_a[i]*nrm_r[i]))
                cond_s2++;
        }

        /** 8. check convergence */
        if (cond_s1 == k) {
            if (log_lev1)
                params.log_stream << params.prefix
                                  << "LSQR: Convergence (S1)!" << std::endl;
            return -2;
        }

        if (cond_s2 == k) {
            if (log_lev1)
                params.log_stream << params.prefix
                                  << "LSQR: Convergence (S2)!" << std::endl;
            return -3;
        }

        /** 9. estimate of cond(A) */
        base::ColumnNrm2(W, nrm_w);
        for (index_type i=0; i<k; ++i) {
            sq_w[i] = nrm_w[i]*nrm_w[i];
            sq_d[i] += sq_w[i]/(rho[i]*rho[i]);
            cnd_a[i] = nrm_a[i]*sqrt(sq_d[i]);

            /** 10. check condition number */
            if (cnd_a[i]>(1.0/eps)) {
                if (log_lev1)
                    params.log_stream << params.prefix
                                      << "LSQR: Stopping (S3)!" << std::endl;
                return -4;
            }
        }

        /** 11. check stagnation */
        for (index_type i=0; i<k; ++i) {
            if (std::abs(phi[i]/rho[i])*nrm_w[i] < (eps*nrm_x[i]))
                stag[i]++;
            else
                stag[i] = 0;

            if (stag[i] >= max_n_stag) {
                if (log_lev1)
                    params.log_stream << params.prefix
                                      << "LSQR: Stagnation." << std::endl;
                return -5;
            }
        }

        /** 12. estimate of norm(X) */
        for (index_type i=0; i<k; ++i) {
            delta[i] =  sn2[i]*rho[i];
            gambar[i] = -cs2[i]*rho[i];
            rhs[i] = phi[i] - delta[i]*z[i];
            zbar[i] = rhs[i]/gambar[i];
            nrm_x[i] = sqrt(sq_x[i] + (zbar[i]*zbar[i]));
            gamma[i] = sqrt((gambar[i]*gambar[i]) + (theta[i]*theta[i]));
            cs2[i] = gambar[i]/gamma[i];
            sn2[i] = theta[i]/gamma[i];
            z[i] = rhs[i]/gamma[i];
            sq_x[i] += z[i]*z[i];
        }
    }
    if (log_lev1)
        params.log_stream << params.prefix
                          << "LSQR: No convergence within iteration limit."
                          << std::endl;

    return -6;
}

} } /** namespace skylark::algorithms */

#endif // SKYLARK_LSQR_HPP
