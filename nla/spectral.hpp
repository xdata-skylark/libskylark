#ifndef SKYLARK_SPECTRAL_HPP
#define SKYLARK_SPECTRAL_HPP

#include <El.hpp>

namespace skylark { namespace nla {

/**
 * Chebyshev points of the second kind.
 *
 * Returns the N Chebyshev points of the second kind, rescaled to [a,b].
 * That is, x_j = (cos(j* pi / N) + a + 1) * (b - a) / 2 for j=0,..N-1.
 */
template<typename T>
void ChebyshevPoints(int N, El::Matrix<T>& X, double a = -1, double b = 1) {
    const double s = (b - a) / 2.0;
    const double pi = boost::math::constants::pi<double>();

    N = N - 1;
    X.Resize(N+1, 1);
    for(int j = 0; j <= N; j++)
        X.Set(j, 0,
            (std::cos(j * pi / N) + a + 1) * s);

    if (N % 2 == 0)
        X.Set(N / 2, 0, 0.0);
}

/**
 * Differentation matrix associated with with interpolation on N Chebyshev
 * points of the second kind.
 *
 * Returns a NxN matrix with the following property:
 *
 * Suppose a vector p representes a polynomial p(x) of degree N-1 by keeping its
 * value at the N points x_j = (cos(j* pi / N) + a + 1) * (b - a) / 2
 * for j=0,..,N-1. That is p_i = p(x_j-1) for i = 1,..,N-1. There is a unique
 * degree N-1 polynomial interpolating those points, and that is p(x).
 *
 * ([a,b] is the range of values of x we are interested in.)
 *
 * The dervitative of p(x), p'(x), is a degree N polynomial as well, and it
 * can be represented in the same manner by a vector p'. D is built such that
 *                    p' = D * p
 *
 * \param N degree of differentation matrix (i.e. degree of polynomials + 1).
 * \param D matrix to be filled.
 * \param X matrix containing the Chebyshev points used.
 * \param a,b  range of the parameter we are interested in.
 */
template<typename T>
void ChebyshevDiffMatrix(int N, El::Matrix<T>& D, El::Matrix<T> &X,
    double a = -1, double b = 1) {

    ChebyshevPoints(N, X);
    N = N - 1;
    double *x = X.Buffer();

    D.Resize(N+1, N+1);
    for(int j = 0; j <= N; j++)
        for(int i = 0; i <= N; i++) {
            int d = i - j;
            double v = 2.0 / (b - a);

            if (i == 0 && j == 0)
                v *= (2.0 * N * N + 1.0) / 6.0;
            else if (i == N && j == N)
                v *= -(2.0 * N * N + 1.0) / 6.0;
            else {
                if (i == 0 || i == N)
                    v *= 2.0;
                if (j ==0 || j == N)
                    v /= 2.0;

                if (d == 0)
                    v *= -x[j] / (2.0 * (1 - x[j] * x[j]));
                else if (d % 2 == 0)
                    v *= 1.0 / (x[i] - x[j]);
                else
                    v *= -1.0 / (x[i] - x[j]);
            }

            D.Set(i, j, v);
        }

    // Rescale points from [-1, 1] to [a, b]
    if (a != -1 && b != 1)
        for(int i = 0; i <= N; i++)
            x[i] = a + (x[i] + 1.0) * (b - a) / 2.0;
}

} }

#endif
