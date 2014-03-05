#ifndef SKYLARK_FJLT_DATA_HPP
#define SKYLARK_FJLT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "RFUT_data.hpp"
#include "../utility/randgen.hpp"

#include "transform_data.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for FJLT. Essentially, it
 * holds the input and sketched matrix sizes, the vector of samples
 * and the data of the underlying transform.
 */
template <typename ValueType>
struct FJLT_data_t : public transform_data_t {
    // Typedef value, distribution and data types so that we can use them
    // regularly and consistently
    typedef ValueType value_type;
    typedef boost::random::uniform_int_distribution<int>
    value_distribution_type;
    typedef utility::rademacher_distribution_t<ValueType>
    underlying_value_distribution_type;
    typedef RFUT_data_t<value_type,
                        underlying_value_distribution_type>
        underlying_data_type;
    typedef transform_data_t base_t;

    /**
     * Regular constructor
     */
    FJLT_data_t (int N, int S, skylark::sketch::context_t& context)
        : base_t(N, S, context, "FJLT"),
          samples(base_t::_S),
          underlying_data(base_t::_N, base_t::_context) {

        _populate();
    }

    FJLT_data_t (boost::property_tree::ptree &json,
                 skylark::sketch::context_t& context)
        : base_t(json, context),
          samples(base_t::_S),
          underlying_data(base_t::_N, base_t::_context) {

        _populate();
    }

    const FJLT_data_t& get_data() const {
        return static_cast<const FJLT_data_t&>(*this);
    }


protected:
    std::vector<int> samples; /**< Vector of samples */
    const underlying_data_type underlying_data;
    /**< Data of the underlying RFUT transformation */


    void _populate() {

        value_distribution_type distribution(0, base_t::_N - 1);
        samples = _context.generate_random_samples_array(base_t::_S, distribution);
    }

  };

} } /** namespace skylark::sketch */

#endif /** SKYLARK_FJLT_DATA_HPP */
