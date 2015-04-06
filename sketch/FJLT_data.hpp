#ifndef SKYLARK_FJLT_DATA_HPP
#define SKYLARK_FJLT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/distributions.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for FJLT. Essentially, it
 * holds the input and sketched matrix sizes, the vector of samples
 * and the data of the underlying transform.
 */
struct FJLT_data_t : public sketch_transform_data_t {

    typedef boost::random::uniform_int_distribution<size_t>
    value_distribution_type;
    typedef utility::rademacher_distribution_t<double>
    underlying_value_distribution_type;

    typedef sketch_transform_data_t base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

    };

    FJLT_data_t (int N, int S, base::context_t& context)
        : base_t(N, S, context, "FJLT"),
          samples(base_t::_S) {

        context = build();
    }

    FJLT_data_t (int N, int S, const params_t& params, base::context_t& context)
        : base_t(N, S, context, "FJLT"),
          samples(base_t::_S) {

        context = build();
    }

    FJLT_data_t (const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "FJLT"),
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
    virtual sketch_transform_t<boost::any, boost::any> *get_transform();

protected:

    FJLT_data_t (int N, int S, const base::context_t& context,
        std::string type)
        : base_t(N, S, context, type),
          samples(base_t::_S) {
    }

    base::context_t build() {
        base::context_t ctx = base_t::build();
        underlying_data = boost::shared_ptr<underlying_data_type>(new
            underlying_data_type(base_t::_N, ctx));
        value_distribution_type distribution(0, base_t::_N - 1);
        samples = ctx.generate_random_samples_array(base_t::_S, distribution);
        return ctx;
    }

    typedef RFUT_data_t<underlying_value_distribution_type>
        underlying_data_type;

    std::vector<size_t> samples; /**< Vector of samples */
    boost::shared_ptr<underlying_data_type> underlying_data;
    /**< Data of the underlying RFUT transformation */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_FJLT_DATA_HPP */
