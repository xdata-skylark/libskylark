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
 * The method is normally applied to SPD matrix A. However, you can
 * attempt to apply it even for a non-symmetric matrix, but be aware
 * that the code will operate actually on A^T in that case.
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
void AsyRGS(const base::sparse_matrix_t<T1>& A, const elem::Matrix<T2>& B,
    elem::Matrix<T3>& X, int sweeps, base::context_t& context) {

    int n = A.height();   // We assume A is square. TODO assert it.

    const int *colptr = A.indptr();
    const int *rowind = A.indices();
    const T1 *vals = A.locked_values();

    typedef boost::random::uniform_int_distribution<int> dtype;
    dtype distribution(0, n-1);
    utility::random_samples_array_t<dtype> stepidxs =
        context.allocate_random_samples_array(sweeps * n, distribution);

    if (B.Width() == 1) {
       int j;

       const T2 *Bd = B.LockedBuffer();
       T3 *Xd = X.Buffer();

#       pragma omp parallel for default(shared) private(j)
        for(j = 0; j < sweeps * n ; j++)
            internal::jstep1(colptr, rowind, vals, Bd, Xd, stepidxs[j]);

    } else {
        elem::Matrix<T2> BT;
        elem::Transpose(B, BT);
        elem::Matrix<T3> XT;
        elem::Transpose(X, XT);

        const T2 *Bd = BT.LockedBuffer();
        T3 *Xd = XT.Buffer();

        int k = B.Width();
        T3 d[k];

        int j;

#       pragma omp parallel for default(shared) private(j, d)
        for(j = 0; j < sweeps * n ; j++)
            internal::jstep(colptr, rowind, vals, Bd, Xd, k, d, stepidxs[j]);

        elem::Transpose(XT, X);
    }
}

} } // namespace skylark::algorithms

#endif  // if SKYLARK_HAVE_ELEMENTAL && SKYLARK_HAVE_OPENMP

#endif
