#ifndef RFUT_DATA_HPP
#define RFUT_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/randgen.hpp"

#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for RFUT transform. Essentially, it
 * holds the input and sketched matrix sizes and array D[].
 */
template <typename ValueType,
          typename ValueDistributionType>
struct RFUT_data_t {
    typedef ValueType value_type;
    typedef ValueDistributionType value_distribution_type;

    /**
     * Regular constructor
     */
    RFUT_data_t (int N, skylark::sketch::context_t& context)
        : N(N), context(context), D(N) {
        value_distribution_type distribution;
        skylark::utility::random_samples_array_t
            <value_type, value_distribution_type>
            random_samples =
            context.allocate_random_samples_array
            <value_type, value_distribution_type>
            (N, distribution);
        for (int i = 0; i < N; i++) {
            D[i] = random_samples[i] ? +1 : -1;
        }
    }

    RFUT_data_t& get_data() {
        return static_cast<RFUT_data_t&>(*this);
    }


protected:

    const int N; /**< Input dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    std::vector<value_type> D;
  };

} } /** namespace skylark::sketch */

#endif /** RFUT_DATA_HPP */
