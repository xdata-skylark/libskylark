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
struct PPT_data_t : public sketch_transform_data_t {

    typedef sketch_transform_data_t base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(int q = 3, double c = 1.0, double gamma = 1.0) :
            q(q), c(c), gamma(gamma) {

        }
        const int q;
        const double c;
        const double gamma;
    };

    PPT_data_t (int N, int S, int q, double c, double gamma,
                base::context_t& context)
        : base_t(N, S, context, "PPT"), _q(q), _c(c), _gamma(gamma) {

        context = build();
    }

    PPT_data_t (int N, int S, const params_t& params,
                base::context_t& context)
        : base_t(N, S, context, "PPT"), _q(params.q), _c(params.c),
          _gamma(params.gamma) {

        context = build();
    }

    PPT_data_t (const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "PPT"),
        _q(pt.get<int>("q")),
        _c(pt.get<double>("c")),
        _gamma(pt.get<double>("gamma")) {

    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual
    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        pt.put("q", _q);
        pt.put("c", _c);
        pt.put("gamma", _gamma);
        return pt;
    }

protected:

    PPT_data_t (int N, int S, int q, double c, double gamma,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type), _q(q), _c(c), _gamma(gamma) {


    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        for(int i = 0; i < _q; i++)
            _cwts_data.push_back(
                 CWT_data_t(base_t::_N, base_t::_S, ctx));

        boost::random::uniform_int_distribution<size_t>
            distidx(0, base_t::_S - 1);
        _hash_idx = ctx.generate_random_samples_array(_q, distidx);

        utility::rademacher_distribution_t<double> distval;
        _hash_val = ctx.generate_random_samples_array(_q, distval);

        return ctx;
    }

    const int _q;         /**< Polynomial degree */
    const double _c;
    const double _gamma;

    // Hashing info for the homogenity parameter c
    std::vector<size_t> _hash_idx;
    std::vector<double> _hash_val;

    std::list< CWT_data_t > _cwts_data;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_PPT_DATA_HPP */
