#ifndef SKYLARK_UST_DATA_HPP
#define SKYLARK_UST_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/distributions.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for UST (Uniform Sampling Transform). 
 * Essentially, it holds the vector of samples.
 */
struct UST_data_t : public sketch_transform_data_t {

    typedef boost::random::uniform_int_distribution<size_t>
    value_distribution_type;

    typedef sketch_transform_data_t base_t;

    /// Params structure
    struct params_t : public sketch_params_t {
        params_t(bool replace) : replace(replace) {

        }

        const double replace;
    };

    UST_data_t (int N, int S, bool _replace, base::context_t& context)
        : base_t(N, S, context, "UST"),
          _samples(base_t::_S), _replace(_replace) {

        context = build();
    }

    UST_data_t (int N, int S, const params_t& params, base::context_t& context)
        : base_t(N, S, context, "UST"),
          _samples(base_t::_S), _replace(params.replace) {

        context = build();
    }

    UST_data_t (const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "UST"),
        _samples(base_t::_S), _replace(pt.get<bool>("replace")) {

         build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        pt.put("replace", _replace);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const;

protected:

    UST_data_t (int N, int S, bool replace, const base::context_t& context,
        std::string type)
        : base_t(N, S, context, type),
          _samples(base_t::_S), _replace(replace) {
    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        if (_replace) {

            value_distribution_type distribution(0, base_t::_N - 1);
            _samples = ctx.generate_random_samples_array(base_t::_S, distribution);

        } else {

            std::vector<size_t> work(base_t::_N);
            for(int i = 0; i < base_t::_N; i++) {
                boost::random::uniform_int_distribution<int> d(0, i);
                int j = ctx.random_value(d);
                work[i] = work[j];
                work[j] = i;
            }
            std::copy(work.begin(), work.begin() + base_t::_S, _samples.begin());

        }
        return ctx;
    }

    std::vector<size_t> _samples; /**< Vector of samples */
    bool _replace;                /**< With replacement or not */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_UST_DATA_HPP */
