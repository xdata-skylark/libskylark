#ifndef SKYLARK_UTILITY_ELBOWS_HPP
#define SKYLARK_UTILITY_ELBOWS_HPP

#include <math.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>

namespace {
template <typename T>
T dnorm_sum(const std::vector<T>& x, const T mu,
        const T sigma, const bool log=true) {
    double c = 1 / sqrt(2 * M_PI);
    T sum = 0;

    for(unsigned i = 0; i < x.size(); ++i) {
        T x_delta = x[i] - mu;
        if (log)
            sum += std::log(exp(-x_delta * x_delta /
                        (2 * sigma * sigma)) * c / sigma);
        else
            sum += exp(-x_delta * x_delta / (2 * sigma * sigma)) * c / sigma;
    }
    return sum;
}

template <typename T>
void mineq(std::vector<T>& v, const T arg, std::vector<T>& ret) {
    if (v.size() != ret.size())
        ret.resize(v.size());

    for (unsigned i = 0; i < v.size(); i++)
        ret[i] = v[i] - arg;
}

template <typename T>
std::vector<T> pluseq(std::vector<T> v, T val, const T delta=0) {
    std::vector<T> ret;
    for (unsigned i = 0; i < v.size(); i++) {
        ret.push_back(v[i] + val);
        val += delta;
    }
    return ret;
}

template <typename T>
void append(std::vector<T>& to, std::vector<T> from) {
    for (typename std::vector<T>::iterator it = from.begin();
            it != from.end(); ++it)
        to.push_back(*it);
}

template <typename T>
void vecpow_inplace(std::vector<T>& v, const T p) {
    for (unsigned i = 0; i < v.size(); i++) {
        if (p == 2)
            v[i]*=v[i];
        else
            v[i] = pow(v[i], p);
    }
}
} // Annoymous namespace

namespace skylark { namespace utility {
template <typename T>
std::vector<unsigned> get_elbows(std::vector<T>& d, const unsigned n=3) {
    std::sort(d.begin(), d.end(), std::greater<int>());
    unsigned p = d.size();

    if (!p)
        throw std::runtime_error("d arg is empty");

    std::vector<T> lq;
    lq.assign(p, 0); // log likelihood function of q

    std::vector<T> frontd, restd;
    for (unsigned q = 0; q < p; q++) {
        frontd.resize(q+1);
        std::copy(d.begin(), d.begin()+(q+1), frontd.begin());

        T mu1 = std::accumulate(frontd.begin(), frontd.end(), (T)0) /
            (T)frontd.size();
        restd.resize(d.size() - (q+1));
        std::copy(d.begin()+(q+1), d.end(), restd.begin());

        T mu2 = 0;
        if (q + 1 < p) // Avoid div 0
            mu2 = (T)std::accumulate(restd.begin(), restd.end(), (T)0) /
                restd.size();

        std::vector<T> lhsigma, rhsigma;
        mineq(frontd, mu1, lhsigma);
        mineq(restd, mu2, rhsigma);
        vecpow_inplace(lhsigma, (T)2);
        vecpow_inplace(rhsigma, (T)2);

        T denom = (p - 1 - ((q+1) < p ? 1 : 0));

        if (denom) {
            T sigma = (std::accumulate(lhsigma.begin(), lhsigma.end(), (T)0) +
                    std::accumulate(rhsigma.begin(), rhsigma.end(), (T)0)) /
                    denom;

        T sqrtsigma = sqrt(sigma);
        lq[q] = dnorm_sum(frontd, mu1, sqrtsigma) +
            dnorm_sum(restd, mu2, sqrtsigma);
        } else {
            lq[q] = std::numeric_limits<T>::min();
        }
    }

    std::vector<unsigned> q;
    // Index of max
    q.push_back(std::max_element(lq.begin(), lq.end()) - lq.begin());

    if (n > 1 && q.back() < p) {
        std::vector<T> newd(d.size() - (q.back()+1));
        std::copy(&d[q.back()+1], &d[p], newd.begin());
        append(q, pluseq<unsigned>(get_elbows<T>(newd, n-1), q.back()));
    }
    // Adjust for 0-based indexing in C
    return pluseq<unsigned>(q, 1);
}
} } // namespace skylark::utility

#endif // SKYLARK_UTILITY_ELBOWS_HPP
