#ifndef FJLT_DATA_HPP
#define FJLT_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/randgen.hpp"

#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for RFUT transform. Essentially, it
 * holds the input and sketched matrix sizes and array D[].
 */
template <typename ValueType>
struct FJLT_data_t {
    typedef ValueType value_type;
    typedef boost::random::uniform_int_distribution<int>
    value_distribution_type;
    typedef utility::rademacher_distribution_t<ValueType>
    underlying_value_distribution_type;
    typedef RFUT_data_t<value_type, underlying_value_distribution_type>
    underlying_data_type;

    /**
     * Regular constructor
     */
    FJLT_data_t (int N, int S, skylark::sketch::context_t& context)
        : N(N), S(S), context(context),
          samples(S),
          underlying_data(N, context) {
        value_distribution_type distribution(0, N-1);
        skylark::utility::random_samples_array_t
            <value_type, value_distribution_type>
            random_samples =
            context.allocate_random_samples_array
            <value_type, value_distribution_type>
            (S, distribution);
        for (int i = 0; i < S; i++) {
            samples[i] = random_samples[i];
        }
    }

    FJLT_data_t& get_data() {
        return static_cast<FJLT_data_t&>(*this);
    }


protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    std::vector<value_type> samples;
    underlying_data_type underlying_data;
  };

} } /** namespace skylark::sketch */

#endif /** FJLT_DATA_HPP */
