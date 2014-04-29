#ifndef SKYLARK_LSQR_HPP
#define SKYLARK_LSQR_HPP

#include "../base/base.hpp"
#include "../utility/typer.hpp"
#include "../utility/external/print.hpp"


namespace skylark { namespace utility {

template<typename ET>
struct elem_extender_t : public ET {

    // Once we have c'tor inheritance (e.g. gcc-4.8) we can simply
    // use the following line:
    // using ET::ET;
    // For now, I am just implementing the most basic c'tors. More
    // will be added as neccessary.
    elem_extender_t(int m, int n) : ET(m, n) { }

private:
    typedef typename utility::typer_t<ET>::value_type value_type;

public:
    value_type &operator[](int i) {
        return *(ET::Buffer() + i);
    }

    const value_type &operator[](int i) const {
        return *(ET::Buffer() + i);
    }
};

} }
namespace skylark { namespace nla {



/**
 * LSQR method.
 *
 * X should be allocated, but we zero it on start. (not set as X_0).
 */
template<typename MatrixType, typename RhsType, typename SolType>
void LSQR(const MatrixType& A, const RhsType& B, SolType& X,
    iter_params_t params = iter_params_t()) {

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef RhsType rhs_type;        // Also serves as "long" vector type.
    typedef SolType sol_type;        // Also serves as "short" vector type.

    typedef utility::elem_extender_t<
        elem::DistMatrix<value_t, elem::STAR, elem::STAR> >
        scalar_cont_type;

    /** Throughout, we will use m, n, k to denote the problem dimensions */
    index_t m = base::Height(A);
    index_t n = base::Width(A);
    index_t k = base::Width(B);

    /** Set the parameter values accordingly */
    const value_t eps = 32*std::numeric_limits<value_t>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    /** Initialize everything */
    rhs_type U(B);
    scalar_cont_type beta(k, 1), i_beta(k, 1);  // TODO: correct grid!
    base::ColumnNrm2(U, beta);
    for (index_t i=0; i<k; ++i)
        i_beta[i] = 1 / beta[i];
    base::DiagonalScale(elem::RIGHT, elem::NORMAL, i_beta, U);
    //print_vec_t::apply(U,"U Init", params.am_i_printing, params.debug_level); // TODO

    sol_type V(X);     // No need to really copy, just want sizes&comm correct.
    base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, A, U, V);
    scalar_cont_type alpha (k, 1), i_alpha(k, 1);
    base::ColumnNrm2(V, alpha);
    for (index_t i=0; i<k; ++i)
        i_alpha[i] = 1 / alpha[i];
    base::DiagonalScale(elem::RIGHT, elem::NORMAL, i_alpha, V);
    //print_vec_t::apply(U,"V Init", params.am_i_printing, params.debug_level); // TODO

    /* Create W=V and X=0 */
    base::Zero(X);
    sol_type W(V);
    scalar_cont_type phibar(beta), rhobar(alpha), nrm_r(beta);
    scalar_cont_type nrm_a(k, 1), cnd_a(k, 1), sq_d(k, 1), nrm_ar_0(k, 1);
    base::Zero(nrm_a); base::Zero(cnd_a); base::Zero(sq_d);
    elem::Hadamard(alpha, beta, nrm_ar_0);

    /** Return from here */
    for (index_t i=0; i<k; ++i) if (nrm_ar_0[i]==0) {
            params.return_code=0;
            return;
        }

    scalar_cont_type nrm_x(k, 1), sq_x(k, 1), z(k, 1), cs2(k, 1), sn2(k, 1);
    elem::Zero(nrm_x); elem::Zero(sq_x); elem::Zero(z); elem::Zero(sn2);
    for (index_t i=0; i<k; ++i) 
        cs2[i] = -1.0;

    int max_n_stag = 3;
    std::vector<int> stag(k, 0);

    /* Reset the iteration limit if none was specified */
    if (0>params.iter_lim) params.iter_lim = std::max(20, 2*std::min(m,n));

    /* More varaibles */
    rhs_type AV(B);
    sol_type AU(X);
    scalar_cont_type minus_beta(k, 1);
    scalar_cont_type rho(k,1);
    scalar_cont_type cs(k, 1), sn(k, 1), theta(k, 1), phi(k, 1);
    scalar_cont_type phi_by_rho(k, 1), minus_theta_by_rho(k, 1), nrm_ar(k, 1);
    scalar_cont_type nrm_w(k, 1), sq_w(k, 1), gamma(k, 1);
    scalar_cont_type delta(k, 1), gambar(k, 1), rhs(k, 1), zbar(k, 1);

    /** Main iteration loop */
    for (index_t itn=0; itn<params.iter_lim; ++itn) {

        /** 1. Update u and beta */
        base::Gemm(elem::NORMAL, elem::NORMAL, 1.0, A, V, AV);
        elem::Scal(-1.0, alpha);   // Can safely overwrite based on subseq ops.
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, alpha, U);
        base::Axpy(1.0, AV, U);
        base::ColumnNrm2(U, beta);
        for (index_t i=0; i<k; ++i)
            i_beta[i] = 1 / beta[i];
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, i_beta, U);

        /** 2. Estimate norm of A */
        for (index_t i=0; i<k; ++i) {
            double a = nrm_a[i], b = alpha[i], c = beta[i];
            nrm_a[i] = sqrt(a*a + b*b + c*c);
        }

        /** 3. Update v */
        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, A, U, AU);
        for (index_t i=0; i<k; ++i)
            minus_beta[i] = -beta[i];
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, minus_beta, V);
        base::Axpy(1.0, AU, V);
        base::ColumnNrm2(V, alpha);
        for (index_t i=0; i<k; ++i)
            i_alpha[i] = 1 / alpha[i];
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, i_alpha, V);

        /** 4. Define some variables */
        for (index_t i=0; i<k; ++i) {
            rho[i] = sqrt((rhobar[i]*rhobar[i]) + (beta[i]*beta[i]));
            cs[i] = rhobar[i]/rho[i];
            sn[i] =  beta[i]/rho[i];
            theta[i] = sn[i]*alpha[i];
            rhobar[i] = -cs[i]*alpha[i];
            phi[i] = cs[i]*phibar[i];
            phibar[i] =  sn[i]*phibar[i];
        }

        /** 5. Update X and W */
        for (index_t i=0; i<k; ++i)
            phi_by_rho[i] = phi[i]/rho[i];
        base::Axpy(phi_by_rho, W, X);
        //print_vec_t::apply(X, "X", params.am_i_printing, params.debug_level);

        for (index_t i=0; i<k; ++i)
            minus_theta_by_rho[i] = -theta[i]/rho[i];
        base::DiagonalScale(elem::RIGHT, elem::NORMAL, minus_theta_by_rho, W);
        base::Axpy(1.0, V, W);
        //print_vec_t::apply(W, "W", params.am_i_printing, params.debug_level);

        /** 6. Estimate norm(r) */
        nrm_r = phibar;

        /** 7. estimate of norm(A'*r) */
        for (index_t i=0; i<k; ++i) {
            nrm_ar[i] = phibar[i]*alpha[i]*std::abs(cs[i]);

            /** 8. check convergence */
            if (nrm_ar[i]<(params.tolerance*nrm_ar_0[i])) {
                params.return_code = -2;
                return;
            }
            if (nrm_ar[i]<(eps*nrm_a[i]*nrm_r[i])) {
                params.return_code = -3;
                return;
            }
        }

        /** 9. estimate of cond(A) */
        base::ColumnNrm2(W, nrm_w); 
        for (index_t i=0; i<k; ++i) {
            sq_w[i] = nrm_w[i]*nrm_w[i];
            sq_d[i] += sq_w[i]/(rho[i]*rho[i]);
            cnd_a[i] = nrm_a[i]*sqrt(sq_d[i]);

            /** 10. check condition number */
            if (cnd_a[i]>(1.0/eps)) { params.return_code = -4; return; }
        }

        /** 11. check stagnation */
        for (index_t i=0; i<k; ++i) {
            if (std::abs(phi[i]/rho[i])*nrm_w[i] < (eps*nrm_x[i]))
                stag[i]++;
            else
                stag[i] = 0;

            if (stag[i] >= max_n_stag) {
                params.return_code = -5;
                return;
            }
        }

        /** 12. estimate of norm(X) */
        for (index_t i=0; i<k; ++i) {
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
    params.return_code = -6;
}

} } /** namespace skylark::nla */

#endif // SKYLARK_LSQR_HPP
