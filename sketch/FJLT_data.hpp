#ifndef SKYLARK_FJLT_DATA_HPP
#define SKYLARK_FJLT_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "RFUT_data.hpp"
#include "../utility/randgen.hpp"

#include "sketch_transform_data.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for FJLT. Essentially, it
 * holds the input and sketched matrix sizes, the vector of samples
 * and the data of the underlying transform.
 */
template <typename ValueType>
struct FJLT_data_t : public sketch_transform_data_t {
    // Typedef value, distribution and data types so that we can use them
    // regularly and consistently
    typedef ValueType value_type;
    typedef boost::random::uniform_int_distribution<int>
    value_distribution_type;
    typedef utility::rademacher_distribution_t<ValueType>
    underlying_value_distribution_type;


    typedef sketch_transform_data_t base_t;

    /**
     * Regular constructor
     */
    FJLT_data_t (int N, int S, skylark::base::context_t& context)
        : base_t(N, S, context, "FJLT"),
          samples(base_t::_S), underlying_data(nullptr) {

        context = build();
    }

    FJLT_data_t (boost::property_tree::ptree &json)
        : base_t(json),
          samples(base_t::_S) {

         build();
    }

    virtual ~FJLT_data_t() {
        delete underlying_data;
    }

protected:

    FJLT_data_t (int N, int S, skylark::base::context_t& context, 
        std::string type)
        : base_t(N, S, context, type),
          samples(base_t::_S), underlying_data(nullptr) {
    }

    FJLT_data_t (boost::property_tree::ptree &json, bool nobuild)
        : base_t(json),
          samples(base_t::_S) {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();
        underlying_data = new underlying_data_type(base_t::_N, ctx);
        value_distribution_type distribution(0, base_t::_N - 1);
        samples = ctx.generate_random_samples_array(base_t::_S, distribution);
        return ctx;
    }

    typedef RFUT_data_t<value_type,
                        underlying_value_distribution_type>
        underlying_data_type;

    std::vector<int> samples; /**< Vector of samples */
    underlying_data_type *underlying_data;
    /**< Data of the underlying RFUT transformation */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_FJLT_DATA_HPP */
