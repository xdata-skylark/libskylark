#ifndef SKYLARK_QUASIRAND_HPP
#define SKYLARK_QUASIRAND_HPP

#include "boost/property_tree/ptree.hpp"
#include "boost/math/special_functions/prime.hpp"

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

template<typename ValueType> 
struct leaped_halton_sequence_t {

    typedef ValueType value_type;

    leaped_halton_sequence_t() :
        _d(0), _leap(0) {
    }

    leaped_halton_sequence_t(size_t d, size_t leap = -1) :
        _d(d),
        _leap(leap == -1 ? boost::math::prime(d) : leap) {

    }

    leaped_halton_sequence_t (const boost::property_tree::ptree& json) {
        _d = json.get<int>("d");
        _leap = json.get<size_t>("leap");
    }

    leaped_halton_sequence_t& operator=(const leaped_halton_sequence_t& other) {
        _d = other._d;
        _leap = other._leap;
        return *this;
    }

    inline value_type coordinate(size_t idx, size_t i) const {
        return RadialInverseFunction(boost::math::prime(i), idx * _leap);
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "qmc_sequence");
        pt.put("skylark_version", VERSION);
        pt.put("sequence_type", "leaped halton");
        pt.put("d", _d);
        pt.put("leap", _leap);
        return pt;
    }

private:
    size_t _d;
    size_t _leap;
};

} } // namespace skylark::utility

#endif
