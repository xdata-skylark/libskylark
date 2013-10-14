#ifndef SKYLARK_FJLT_DATA_HPP
#define SKYLARK_FJLT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "RFUT_data.hpp"
#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for FJLT. Essentially, it
 * holds the input and sketched matrix sizes, the vector of samples
 * and the data of the underlying transform.
 */
template <typename ValueType>
struct FJLT_data_t {
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

    /**
     * Regular constructor
     */
    FJLT_data_t (int N, int S, skylark::sketch::context_t& context)
        : N(N), S(S), context(context),
          samples(S),
          underlying_data(N, context) {
        value_distribution_type distribution(0, N-1);
        samples = context.generate_random_samples_array
            <value_type, value_distribution_type>
            (S, distribution);
    }


    const FJLT_data_t& get_data() const {
        return static_cast<const FJLT_data_t&>(*this);
    }


protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    std::vector<value_type> samples; /**< Vector of samples */
    const underlying_data_type underlying_data;
    /**< Data of the underlying RFUT transformation */
  };

} } /** namespace skylark::sketch */

#endif /** SKYLARK_FJLT_DATA_HPP */
