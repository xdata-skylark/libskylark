#ifndef SKYLARK_PPT_DATA_HPP
#define SKYLARK_PPT_DATA_HPP

#include <vector>

#include "../utility/distributions.hpp"

#include "../base/context.hpp"
#include "CWT_data.hpp"
#include "transform_data.hpp"

namespace skylark { namespace sketch {

/**
 * Pham-Pagh Transform aka TensorSketch (data).
 *
 * Sketches the monomial expansion of a vector.
 *
 * See:
 * Ninh Pham and Rasmus Pagh
 * Fast and Scalable Polynomial Kernels via Explicit Feature Maps
 * KDD 2013
 */
template <typename ValueType>
struct PPT_data_t : public transform_data_t {

    typedef transform_data_t base_t;

    /**
     * Regular constructor
     */
    PPT_data_t (int N, int S, int q, double c, double gamma,
                skylark::base::context_t& context)
        : base_t(N, S, context, "PPT"), _q(q), _c(c), _gamma(gamma) {

        _populate();
    }

    PPT_data_t (boost::property_tree::ptree &json)
        : base_t(json),
        _q(json.get<int>("sketch.q")),
        _c(json.get<double>("sketch.c")),
        _gamma(json.get<double>("sketch.gamma")) {

        _populate();
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const PPT_data_t<ValueT> &data);
protected:

    typedef CWT_data_t<size_t, ValueType> _CWT_data_t;

    const int _q;         /**< Polynomial degree */
    const double _c;
    const double _gamma;

    // Hashing info for the homogenity parameter c
    std::vector<int> _hash_idx;
    std::vector<double> _hash_val;

    std::list< _CWT_data_t > _cwts_data;

    void _populate() {

        for(int i = 0; i < _q; i++)
            _cwts_data.push_back(
                _CWT_data_t(base_t::_N, base_t::_S, base_t::_creation_context));

        boost::random::uniform_int_distribution<int> distidx(0, base_t::_S - 1);
        _hash_idx = base_t::_creation_context->generate_random_samples_array(_q, distidx);

        utility::rademacher_distribution_t<double> distval;
        _hash_val = base_t::_creation_context->generate_random_samples_array(_q, distval);
    }
};

template <typename ValueType>
boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const PPT_data_t<ValueType> &data) {

    sk << static_cast<const transform_data_t&>(data);
    sk.put("sketch.q", data._q);
    sk.put("sketch.c", data._c);
    sk.put("sketch.gamma", data._gamma);
    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_PPT_DATA_HPP */
