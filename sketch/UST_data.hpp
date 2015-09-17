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

    };

    UST_data_t (int N, int S, base::context_t& context)
        : base_t(N, S, context, "UST"),
          samples(base_t::_S) {

        context = build();
    }

    UST_data_t (int N, int S, const params_t& params, base::context_t& context)
        : base_t(N, S, context, "UST"),
          samples(base_t::_S) {

        context = build();
    }

    UST_data_t (const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "UST"),
          samples(base_t::_S) {

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
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const;

protected:

    UST_data_t (int N, int S, const base::context_t& context,
        std::string type)
        : base_t(N, S, context, type),
          samples(base_t::_S) {
    }

    base::context_t build() {
        base::context_t ctx = base_t::build();
        value_distribution_type distribution(0, base_t::_N - 1);
        samples = ctx.generate_random_samples_array(base_t::_S, distribution);
        return ctx;
    }

    std::vector<size_t> samples; /**< Vector of samples */

};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_UST_DATA_HPP */
