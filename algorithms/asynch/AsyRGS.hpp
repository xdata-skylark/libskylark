#ifndef SKYLARK_ASYRGS_HPP
#define SKYLARK_ASYRGS_HPP

#if SKYLARK_HAVE_OPENMP

namespace skylark {
namespace algorithms {

namespace internal {

template<typename T1, typename T2, typename T3>
inline void jstep(const int *colptr, const int *rowind, const T1 *vals,
    T2 *B, T3 *X, int k, T3 *xvals, int i) {

    double diag = 1.0, v;

    if (colptr[i] == colptr[i+1])
        return;

    for(int r = 0; r < k; r++)
        xvals[r] = B[i * k + r];

    for(int j = colptr[i]; j < colptr[i + 1]; j++) {
        if (rowind[j] == i)
            diag = vals[j];
        v = vals[j];
        double *xx = X + rowind[j] * k;
        for (int r = 0; r < k; r++)
            xvals[r] -= v * xx[r];
    }

    for(int r = 0; r < k; r++) {
        int idx = i * k + r;
        v = xvals[r] / diag;
#       pragma omp atomic
        X[idx] += v;
    }
}

template<typename T1, typename T2, typename T3>
inline void jstep1(const int *colptr, const int *rowind, const T1 *vals,
    T2 *b, T3 *x, int i) {

    double diag = 1.0, v;

    if (colptr[i] == colptr[i+1])
        return;

    v = b[i];
    for(int j = colptr[i]; j < colptr[i + 1]; j++) {
        if (rowind[j] == i)
            diag = vals[j];
        v -= vals[j] * x[rowind[j]];
    }

    v /= diag;
#   pragma omp atomic
    x[i] += v;
}


} // namespace internal

/**
 * Asynchronous Randomized Gauss-Seidel for solving A * X = B.
 *
 * The method is normally applied to HPD matrix A. However, you can
 * attempt to apply it even for a non-Hermitian matrix, but be aware
 * that the code will operate actually on A^* in that case.
 *
 * Reference:
 * Avron, Druinsky and Gupta
 * Revisiting Asynchronous Linear Solvers:
 * Provable Convergence Rate Through Randomization
 * IPDPS 2014
 *
 * @param A input matrix
 * @param B right hand side.
 * @param X output - must be preallocated. The content is used as initial X.
 */
template<typename T1, typename T2, typename T3>
int AsyRGS(const base::sparse_matrix_t<T1>& A, const El::Matrix<T2>& B,
    El::Matrix<T3>& X, base::context_t& context,
    asy_iter_params_t params = asy_iter_params_t()) {

    int ret;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    int n = A.height();   // We assume A is square. TODO assert it.

    const int *colptr = A.indptr();
    const int *rowind = A.indices();
    const T1 *vals = A.locked_values();

    typedef boost::random::uniform_int_distribution<int> dtype;
    dtype distribution(0, n-1);

    if (B.Width() == 1) {
       int j;

       double nrmb = base::Nrm2(B);

       const T2 *Bd = B.LockedBuffer();
       T3 *Xd = X.Buffer();

       int sweeps_left = params.sweeps_lim;
       int done_sweeps = 0;
       while (sweeps_left > 0) {

           int sweeps = params.syn_sweeps > 0 ?
               std::min(params.syn_sweeps, sweeps_left) : sweeps_left;

           base::random_samples_array_t<dtype> stepidxs =
               context.allocate_random_samples_array(sweeps * n, distribution);

#          pragma omp parallel for default(shared) private(j)
           for(j = 0; j < sweeps * n ; j++)
               internal::jstep1(colptr, rowind, vals, Bd, Xd, stepidxs[j]);

           sweeps_left -= sweeps;
           done_sweeps += sweeps;

           if (params.tolerance > 0) {
               El::Matrix<double> R(B);
               base::Gemv(El::ADJOINT, -1.0, A, X, 1.0, R);
               double res = base::Nrm2(R);
               double relres = res / nrmb;

               if (log_lev2)
                   params.log_stream << "AsyRGS: Sweeps = " << done_sweeps
                                     << ", Relres = "
                                     << boost::format("%.2e") % relres << std::endl;

               if(res < nrmb * params.tolerance) {
                   if (log_lev1)
                       params.log_stream << "AsyRGS: Convergence!" << std::endl;
                   ret = -1;
                   goto cleanup;
               }
           }
       }
     } else {
        int j;

        El::Matrix<T2> BT;
        El::Transpose(B, BT);
        El::Matrix<T3> XT;
        El::Transpose(X, XT);

        const T2 *Bd = BT.LockedBuffer();
        T3 *Xd = XT.Buffer();

        int k = B.Width();
        T3 d[k];

        typedef El::Matrix<T2> rhs_type;
        typedef utility::elem_extender_t<
            typename internal::scalar_cont_typer_t<rhs_type>::type >
            scalar_cont_type;
        scalar_cont_type
            nrmb(internal::scalar_cont_typer_t<rhs_type>::build_compatible(k, 1, B));
        double total_nrmb = 0.0;
        if (params.tolerance > 0) {
            base::ColumnNrm2(B, nrmb);
            for(int i = 0; i < k; i++)
                total_nrmb += nrmb[i] * nrmb[i];
        }
        total_nrmb = sqrt(total_nrmb);
        scalar_cont_type ressqr(nrmb);

        int sweeps_left = params.sweeps_lim;
        int done_sweeps = 0;
        while (sweeps_left > 0) {

            int sweeps = params.syn_sweeps > 0 ?
                std::min(params.syn_sweeps, sweeps_left) : sweeps_left;

            base::random_samples_array_t<dtype> stepidxs =
                context.allocate_random_samples_array(sweeps * n, distribution);

#           pragma omp parallel for default(shared) private(j, d)
            for(j = 0; j < sweeps * n ; j++)
                internal::jstep(colptr, rowind, vals, Bd, Xd, k, d, stepidxs[j]);

           sweeps_left -= sweeps;
           done_sweeps += sweeps;

           if (params.tolerance > 0) {

               El::Matrix<double> RT(BT);
               base::Gemm(El::NORMAL, El::NORMAL, -1.0, XT, A, 1.0, RT);
               base::RowDot(RT, RT, ressqr);

               int convg = 0;
               for(int i = 0; i < k; i++) {
                   if (sqrt(ressqr[i]) < (params.tolerance*nrmb[i]))
                       convg++;
               }

               if (log_lev2) {
                   double total_ressqr = 0.0;
                   for(int i = 0; i < k; i++)
                       total_ressqr += ressqr[i];
                   double relres = sqrt(total_ressqr) / total_nrmb;
                   params.log_stream << "AsyRGS: Sweeps = " << done_sweeps
                                     << ", Relres = "
                                     << boost::format("%.2e") % relres
                                     << ", " << convg << " rhs converged" << std::endl;
               }

               if(convg == k) {
                   if (log_lev1)
                       params.log_stream << "AsyRGS: Convergence!" << std::endl;
                   ret = -1;
                   El::Transpose(XT, X);
                   goto cleanup;
               }
           }
        }

        El::Transpose(XT, X);
    }

    ret = -6;
    if (log_lev1)
        params.log_stream << "AsyRGS: No convergence within iteration limit."
                          << std::endl;

 cleanup:

    return ret;
}

} } // namespace skylark::algorithms

#endif  // if SKYLARK_HAVE_OPENMP

#endif
