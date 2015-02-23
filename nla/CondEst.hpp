#ifndef SKYLARK_CONDEST_HPP
#define SKYLARK_CONDEST_HPP

#include <boost/math/special_functions/erf.hpp>

#include "../base/base.hpp"
#include "../utility/typer.hpp"
#include "../utility/external/print.hpp"

extern "C" {

void EL_BLAS(dbdsqr)(const char *, const El::Int *, const El::Int *,
    const El::Int *, const El::Int *, double *, double *, double *,
    const El::Int *, double *, const El::Int *,
    double *, const El::Int *, double *, El::Int *);

}

namespace skylark {
namespace nla {

struct condest_params_t {

    int iter_lim;
    bool am_i_printing;
    int log_level;
    int res_print;
    std::ostream& log_stream;
    int debug_level;

    // See paper for meaning of these.
    int powerits;
    double c1, c2, c3, c4, c1t;

    condest_params_t(int iter_lim = 1000,
        bool am_i_printing = 0,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) : iter_lim(iter_lim),
                               am_i_printing(am_i_printing),
                               log_level(log_level),
                               log_stream(log_stream),
                               debug_level(debug_level) {

        const double em = std::numeric_limits<double>::epsilon();
        c1 = 8 * em; c2 = 1e-3; c3 = 64.0 / em;
        c4 = std::sqrt(em); c1t = 4 * em;
        powerits = 300;
  }

};

/**
 * Estimates the condition number (with certificates) of a matrix
 * using an iterative algorith. Based on the following paper:
 *
 * Haim Avron, Alex Druinsky and Sivan Toledo
 * Spectral Condition-Number Estimation of Large Sparse Matrices
 *
 * \param A Input matrix
 * \param cond Output condition number estimation
 * \param sigma_max,v_max,u_max Estiamte of largest singular value and
 *                               right, left certificates. That is,
 *                               sigma_max * u_max = A * v_max / sigma_max,
 *                               ||v_max|| = ||u_max|| = 1,
 * \param sigma_min Best estimate for smallest singular value.
 * \param sigma_min_c,v_min u_min Estimate of smallest singular value with
 *                                certificate
 * \param context Skylark context
 * \param params Parameters.
 */
template<typename MatrixType, typename LeftType, typename RightType>
int CondEst(const MatrixType& A, double &cond,
    double &sigma_max, RightType &v_max, LeftType &u_max,
    double &sigma_min, double &sigma_min_c, RightType &v_min, LeftType &u_min,
    base::context_t &context, condest_params_t params = condest_params_t()) {

    typedef typename utility::typer_t<MatrixType>::value_type value_t;
    typedef typename utility::typer_t<MatrixType>::index_type index_t;

    typedef MatrixType matrix_type;
    typedef RightType right_type;        // Also serves as "long" vector type.
    typedef LeftType left_type;        // Also serves as "short" vector type.

    typedef utility::print_t<right_type> rhs_print_t;
    typedef utility::print_t<left_type> sol_print_t;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    /** Throughout, we will use m, n to denote the problem dimensions */
    index_t m = base::Height(A);
    index_t n = base::Width(A);

    double c1 = params.c1, c2 = params.c2, c3 = params.c3;
    double c4 = params.c4, c1t = params.c1t;

    /** Estimate the largest singular vector using power-iteration */
    u_max.Resize(m, 1);
    base::GaussianMatrix(v_max, n, 1, context);
    El::Scale(1.0 / El::Nrm2(v_max), v_max);
    for(int i = 0; i < params.powerits; i++) {
        base::Gemm(El::NORMAL, El::NORMAL, 1.0, A, v_max, u_max);
        El::Scale(1.0 / El::Nrm2(u_max), u_max);
        base::Gemm(El::ADJOINT, El::NORMAL, 1.0, A, u_max, v_max);
        El::Scale(1.0 / El::Nrm2(v_max), v_max);
    }
    base::Gemm(El::NORMAL, El::NORMAL, 1.0, A, v_max, u_max);
    sigma_max = El::Nrm2(u_max);
    El::Scale(1.0 / sigma_max, u_max);

    sigma_min = sigma_max;
    u_min = u_max;
    v_min = v_max;

    if (log_lev2)
        params.log_stream << "CondEst: sigma_max = " << sigma_max
                          << std::endl;

    /** Generate xhat, and figure out tau */
    right_type xhat;
    base::GaussianMatrix(xhat, n, 1, context);
    double nrm_xhat = El::Nrm2(xhat);
    double tau = std::sqrt(2) * boost::math::erf_inv(c2) / nrm_xhat;
    El::Scale(1.0 / nrm_xhat, xhat);

    if (log_lev2)
        params.log_stream << "CondEst: tau = " << tau << std::endl;

    /** Generate b, and iteration x */
    left_type b(m, 1);
    base::Gemm(El::NORMAL, El::NORMAL, 1.0, A, xhat, b);
    right_type x(n, 1);
    double nrm_b = El::Nrm2(b);

    /** Initialize everything */
    right_type u(b);
    double beta, i_beta;
    beta = El::Nrm2(u);
    El::Scale(1.0 / beta, u);
    rhs_print_t::apply(u, "u Init", params.am_i_printing, params.debug_level);

    left_type v(n, 1);
    base::Gemm(El::ADJOINT, El::NORMAL, 1.0, A, u, v);
    double alpha;
    alpha = El::Nrm2(v);
    El::Scale(1.0 / alpha, v);
    sol_print_t::apply(v, "v Init", params.am_i_printing, params.debug_level);

    /* Create w=v and x=0 */
    base::Zero(x);
    left_type w(v);
    double phibar = beta, rhobar = alpha, nrm_r;

    /* Reset the iteration limit if none was specified */
    if (0>params.iter_lim) params.iter_lim = std::max(20, 2*std::min(m,n));

    /* More varaibles */
    left_type Au(n, 1);
    double minus_beta, rho;
    double cs, sn, theta, phi;
    double phi_by_rho, minus_theta_by_rho;
    right_type d(n, 1);
    left_type Ad(m, 1);
    std::vector<double> Rdiag, Rsub;

    /** Main iteration loop */
    index_t T = params.iter_lim;
    int retval = -6;
    for (index_t itn=0; itn < T; ++itn) {

        /** 1. Update u and beta */
        alpha = -alpha;
        El::Scale(alpha, u);
        base::Gemm(El::NORMAL, El::NORMAL, 1.0, A, v, 1.0, u);
        beta = El::Nrm2(u);
        El::Scale(1.0 / beta, u);

        /** 2. Update v */
        minus_beta = -beta;
        El::Scale(minus_beta, v);
        base::Gemm(El::ADJOINT, El::NORMAL, 1.0, A, u, Au);
        base::Axpy(1.0, Au, v);
        alpha = El::Nrm2(v);
        El::Scale(1.0 / alpha, v);

       /** 3. Update variables, store parts of R */
        rho = sqrt((rhobar*rhobar) + (beta*beta));

        Rdiag.push_back(rho);
        if (itn > 0)
            Rsub.push_back(theta);

        cs = rhobar/rho;
        sn =  beta/rho;
        theta = sn*alpha;
        rhobar = -cs*alpha;
        phi = cs*phibar;
        phibar =  sn*phibar;

        /** 4. Update x and w */
        phi_by_rho = phi/rho;
        base::Axpy(phi_by_rho, w, x);
        sol_print_t::apply(x, "x", params.am_i_printing, params.debug_level);

        minus_theta_by_rho = -theta/rho;
        El::Scale(minus_theta_by_rho, w);
        base::Axpy(1.0, v, w);
        sol_print_t::apply(w, "w", params.am_i_printing, params.debug_level);

        /** 5. Compute forward error */
        d = xhat;
        El::Axpy(-1.0, x, d);
        double nrm_d = El::Nrm2(d);
        if (nrm_d == 0.0) {
            cond = 1.0;
            sigma_min = sigma_max;
            u_min = u_max;
            v_min = v_max;
            if (log_lev1)
                params.log_stream << "CondEst: Detected condition number 1"
                                  << std::endl;
            return -1;
        }

        /** 6. Compute current estimate of sigma_min using d, and compare it */
        base::Gemm(El::NORMAL, El::NORMAL, 1.0, A, d, Ad);
        double nrm_ad = El::Nrm2(Ad);
        if (nrm_ad <= sigma_min * nrm_d) {
            sigma_min = nrm_ad / nrm_d;
            v_min = d;
            u_min = Ad;
            El::Scale(1.0 / nrm_ad, u_min);
        }

        /** 7. Check if parameters need to be tuned */
        if (c1 != c1t && sigma_min / sigma_max <= c4 ) {
            if (log_lev1)
                params.log_stream
                    << "CondEst: Highly ill-conditioned, C1 adjusted (C4)"
                    << std::endl;
            c1 = c1t;
        }

        /** 8. Test various stopping criteria */
        double nrm_x = El::Nrm2(x);
        if (T == params.iter_lim &&
            nrm_ad <= c1 * (sigma_max * nrm_x + nrm_b)) {
            if (log_lev1)
                params.log_stream << "CondEst: Convergence detected (C1)"
                                  << std::endl;
            T = 1.25 * itn + 1;
            retval = -2;
        }

        if (T == params.iter_lim && nrm_d <= tau) {
            if (log_lev1)
                params.log_stream << "CondEst: Convergence detected (C2)"
                                  << std::endl;
            T = 1.25 * itn + 1;
            retval = -3;
        }

        if (T == params.iter_lim && sigma_max / sigma_min >= c3) {
            if (log_lev1)
                params.log_stream << "CondEst: Singular?, stopping (C3)"
                                  << std::endl;
            T = 1.25 * itn + 1;
            retval = -4;
        }

        if (log_lev2)
            params.log_stream << "CondEst: Iteration " << itn
                              << " sigma_min = " << sigma_min
                              << " cond = " << sigma_max / sigma_min
                              << " nrm_d = " << nrm_d
                              << std::endl;
    }



    if (log_lev1 && retval == -6)
        params.log_stream << "CondEst: No convergence within iteration limit."
                          << std::endl;

    /** Estimate condition number using R */
    El::Int N = Rdiag.size(), izero = 0, ione = 1, info = 0;
    std::vector<double> workspace(4 * N);
    EL_BLAS(dbdsqr)("Upper", &N, &izero, &izero, &izero, &Rdiag[0], &Rsub[0],
        nullptr, &ione, nullptr, &ione, nullptr, &ione, &workspace[0], &info);
    double sigma_min_R = Rdiag[Rdiag.size() - 1];

    if (log_lev2)
        params.log_stream << "CondEst: R sigma_min = " << sigma_min_R
                          << std::endl;

    sigma_min_c = sigma_min;
    if (sigma_min_R < sigma_min)
        sigma_min = sigma_min_R;

    cond = sigma_max / sigma_min;

    return retval;
}

} } /** namespace skylark::nla */

#endif // SKYLARK_NLA_HPP
