#ifndef SKYLARK_LSQR_HPP
#define SKYLARK_LSQR_HPP

#include <numeric>
#include <functional>
#include <vector>

#include "../utility/external/print.hpp"

namespace skylark { namespace nla {

template <typename MatrixType,
          typename MultiVectorType>
struct lsqr_t {
  /** Typedefs for some variables */
  typedef MatrixType matrix_t;
  typedef MultiVectorType multi_vector_t;
  typedef skylark::nla::iter_solver_op_t<MatrixType, MultiVectorType>
                                                                 iter_ops_t;
  typedef skylark::utility::print_t<multi_vector_t> print_vec_t;
  typedef skylark::utility::print_t<matrix_t> print_mat_t;
  typedef typename iter_ops_t::index_type index_t;
  typedef typename iter_ops_t::value_type value_t;

  /**
   * A routine to print all the errors:
   */
  static void print_error (int code) {
    switch (code) {
      case  0: printf ("Execution successful before first iteration\n"); break;
      case -1: printf ("Returned because matrix dims did not match\n"); break;
      case -2: printf ("Returned because nrm_ar < (tol*nrm_ar_o)\n"); break;
      case -3: printf ("Returned because nrm_ar < eps*nrm_a*nrm_r\n"); break;
      case -4: printf ("Returned because cnd_a < 1/eps\n"); break;
      case -5: printf ("Returned because (stag >= max_n_stag)\n"); break;
      case -6: printf ("Returned because (iter >= iter_lim)\n"); break;
      default: printf ("Unknown execution code\n"); break;
    };
  }

  /** Applicator */
  static void apply (const matrix_t& A,
                     const multi_vector_t& B,
                     multi_vector_t& X,
                     iter_params_t params) {

    /** Create an iter_ops_t object for all our troubles */
    iter_ops_t iter_ops;

    /** Assume error checking has happened elsewhere */
    index_t m1, n1, m2, n2, m3, n3, m, n, k;
    iter_ops.get_dim (A, m1, n1);
    iter_ops.get_dim (B, m2, n2);
    iter_ops.get_dim (X, m3, n3);

    if (m1 != m2) { /* error */ params.return_code=-1; return; }
    if (n1 != m3) { /* error */ params.return_code=-1; return; }
    if (n2 != n3) { /* error */ params.return_code=-1; return; }

    /** Throughout, we will use m, n, k to denote the problem dimensions */
    m = m1; n = n1; k = n2;

    /** Set the parameter values accordingly */
    const value_t eps = 32*std::numeric_limits<value_t>::epsilon();
    if (params.tolerance<eps) params.tolerance=eps;
    else if (params.tolerance>=1.0) params.tolerance=(1-eps);
    else {} /* nothing */

    /** Initialize everything */
    matrix_t AT(iter_ops.transpose(A));
    multi_vector_t U(B);
    std::vector<value_t> beta (k), i_beta(k);
    iter_ops.norm (U, beta);
    for (index_t i=0; i<k; ++i) i_beta[i] = 1.0/beta[i];
    iter_ops.scale (i_beta, U);
    print_vec_t::apply(U,"U Init", params.am_i_printing, params.debug_level);

    multi_vector_t V(n,k);
    iter_ops.mat_vec (AT, U, V);
    std::vector<value_t> alpha (k), i_alpha(k, 1.0);
    iter_ops.norm (V, alpha);
    for (index_t i=0; i<k; ++i) i_alpha[i] = 1.0/alpha[i];
    iter_ops.scale (i_alpha, V);
    print_vec_t::apply(U,"V Init", params.am_i_printing, params.debug_level);

    /* Create W=V and X=0 */
    std::vector<value_t> ZEROS(k,0.0);
    multi_vector_t W(V);
    iter_ops.scale(ZEROS, X);
    std::vector<value_t> phibar(beta), rhobar(alpha), nrm_a(k,0.0),
                         cnd_a(k,0.0), sq_d(k,0.0), nrm_r(beta),
                         nrm_ar_0(k,0.0);
    for (index_t i=0; i<k; ++i) nrm_ar_0[i] = alpha[i]*beta[i];

    /** Return from here */
    for (index_t i=0; i<k; ++i) if (nrm_ar_0[i]==0) {
      params.return_code=0;
      return;
    }

    std::vector<value_t> nrm_x(k,0.0), sq_x(k,0.0), z(k,0.0),
                         cs2(k,-1.0), sn2(k,0);

    value_t max_n_stag = 3;
    std::vector<index_t> stag(k,0);

    /* Reset the iteration limit if none was specified */
    if (0>params.iter_lim) params.iter_lim = std::max(20, 2*std::min(m,n));

    /** Main iteration loop */
    for (index_t itn=0; itn<params.iter_lim; ++itn) {

      /** 1. Update u and beta */
      std::vector<value_t> ONES(k,1.0);
      std::vector<value_t> minus_alpha(k);
      for (index_t i=0; i<k; ++i) minus_alpha[i] = -alpha[i];
      multi_vector_t AV(m, k);
      iter_ops.mat_vec (A, V, AV);
      iter_ops.ax_plus_by (minus_alpha, U, ONES, AV);
      iter_ops.norm (U, beta);
      for (index_t i=0; i<k; ++i) i_beta[i] = 1.0/beta[i];
      iter_ops.scale (i_beta, U);

      /** 2. Estimate norm of A */
      for (index_t i=0; i<k; ++i) nrm_a[i] =
        sqrt((nrm_a[i]*nrm_a[i]) + (alpha[i]*alpha[i]) + (beta[i]*beta[i]));

      /** 3. Update v */
      multi_vector_t AU(n, k);
      std::vector<value_t> minus_beta(k);
      for (index_t i=0; i<k; ++i) minus_beta[i] = -beta[i];
      iter_ops.mat_vec (AT, U, AU);
      iter_ops.ax_plus_by (minus_beta, V, ONES, AU);
      iter_ops.norm (V, alpha);
      for (index_t i=0; i<k; ++i) i_alpha[i] = 1.0/alpha[i];
      iter_ops.scale (i_alpha, V);

      /** 4. Define some variables */
      std::vector<value_t> rho(k), cs(k), sn(k), theta(k), phi(k);
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
      std::vector<value_t> phi_by_rho(k);
      for (index_t i=0; i<k; ++i) phi_by_rho[i] = phi[i]/rho[i];
      iter_ops.ax_plus_by (ONES, X, phi_by_rho, W);
      print_vec_t::apply(X, "X", params.am_i_printing, params.debug_level);

      std::vector<value_t> minus_theta_by_rho(k);
      for (index_t i=0; i<k; ++i) minus_theta_by_rho[i] = -theta[i]/rho[i];
      iter_ops.ax_plus_by(minus_theta_by_rho, W, ONES, V);
      print_vec_t::apply(W, "W", params.am_i_printing, params.debug_level);

      /** 6. Estimate norm(r) */
      for (index_t i=0; i<k; ++i) nrm_r[i] = phibar[i];

      /** 7. estimate of norm(A'*r) */
      std::vector<value_t> nrm_ar(k);
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
      std::vector<value_t> nrm_w(k), sq_w(k);
      iter_ops.norm(W, nrm_w);
      for (index_t i=0; i<k; ++i) {
        sq_w[i] = nrm_w[i]*nrm_w[i];
        sq_d[i] += sq_w[i]/(rho[i]*rho[i]);
        cnd_a[i] = nrm_a[i]*sqrt(sq_d[i]);

        /** 10. check condition number */
        if (cnd_a[i]>(1.0/eps)) { params.return_code = -4; return; }
      }

      /** 11. check stagnation */
      for (index_t i=0; i<k; ++i) {
        if (abs(phi[i]/rho[i])*nrm_w[i] < (eps*nrm_x[i])) stag[i] += 1;
        else stag[i] = 0;

        if (stag[i] >= max_n_stag) { params.return_code = -5; return; }
      }

      /** 12. estimate of norm(X) */
      std::vector<value_t> delta(k), gambar(k), rhs(k), zbar(k), gamma(k);
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
};

} } /** namespace skylark::nla */

#endif // SKYLARK_LSQR_HPP
