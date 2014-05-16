#ifndef SKYLARK_PPT_DATA_HPP
#define SKYLARK_PPT_DATA_HPP

#include <vector>

#include "../utility/distributions.hpp"

#include "../base/context.hpp"
#include "CWT_data.hpp"
#include "sketch_transform_data.hpp"

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
struct PPT_data_t : public sketch_transform_data_t {

    typedef sketch_transform_data_t base_t;

    /**
     * Regular constructor
     */
    PPT_data_t (int N, int S, int q, double c, double gamma,
                skylark::base::context_t& context)
        : base_t(N, S, context, "PPT"), _q(q), _c(c), _gamma(gamma) {

        context = build();
    }

    PPT_data_t (boost::property_tree::ptree &json)
        : base_t(json),
        _q(json.get<int>("sketch.q")),
        _c(json.get<double>("sketch.c")),
        _gamma(json.get<double>("sketch.gamma")) {

    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const PPT_data_t<ValueT> &data);

protected:

    PPT_data_t (int N, int S, int q, double c, double gamma,
        skylark::base::context_t& context, bool nobuild)
        : base_t(N, S, context, "PPT"), _q(q), _c(c), _gamma(gamma) {


    }

    PPT_data_t (boost::property_tree::ptree &json, bool nobuild)
        : base_t(json),
        _q(json.get<int>("sketch.q")),
        _c(json.get<double>("sketch.c")),
        _gamma(json.get<double>("sketch.gamma")) {

    }

    typedef CWT_data_t<size_t, ValueType> _CWT_data_t;

    base::context_t build() {

        base::context_t ctx = base_t::build();

        for(int i = 0; i < _q; i++)
            _cwts_data.push_back(
                 _CWT_data_t(base_t::_N, base_t::_S, ctx));

        boost::random::uniform_int_distribution<int> distidx(0, base_t::_S - 1);
        _hash_idx = ctx.generate_random_samples_array(_q, distidx);

        utility::rademacher_distribution_t<double> distval;
        _hash_val = ctx.generate_random_samples_array(_q, distval);

        return ctx;
    }

    const int _q;         /**< Polynomial degree */
    const double _c;
    const double _gamma;

    // Hashing info for the homogenity parameter c
    std::vector<int> _hash_idx;
    std::vector<double> _hash_val;

    std::list< _CWT_data_t > _cwts_data;
};

template <typename ValueType>
boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const PPT_data_t<ValueType> &data) {

    sk << static_cast<const sketch_transform_data_t&>(data);
    sk.put("sketch.q", data._q);
    sk.put("sketch.c", data._c);
    sk.put("sketch.gamma", data._gamma);
    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_PPT_DATA_HPP */
