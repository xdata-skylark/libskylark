#ifndef SKYLARK_QUASIRAND_HPP
#define SKYLARK_QUASIRAND_HPP

#include <boost/math/special_functions/prime.hpp>

namespace skylark { namespace utility {

inline double RadialInverseFunction(int base, size_t idx)
{
    double r = 0;
    double m = 1.0 / base;
    size_t res = idx + 1;       // We start indexes from 0.
    while(res > 0) {
        r += m * (res % base);
        res /= base;
        m /= base;
    }
    return r;
}

/**
 * Returns the i'th coordinate of the idx's vector of the Halton sequence of
 * dimension d.
 */
inline double Halton(size_t d, size_t idx, size_t i)
{
    return RadialInverseFunction(boost::math::prime(i), idx);
}

} } // namespace skylark::utility

#endif
