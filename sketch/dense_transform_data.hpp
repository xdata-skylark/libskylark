#ifndef SKYLARK_DENSE_TRANSFORM_DATA_HPP
#define SKYLARK_DENSE_TRANSFORM_DATA_HPP

#include <vector>

#include "context.hpp"
#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for dense transforms. Essentially, it
 * holds the input and sketched matrix sizes and the array of samples
 * to be lazily computed.
 */
template <typename ValueType,
          template <typename> class ValueDistribution>
struct dense_transform_data_t {
    // For reasons of naming consistency
    typedef ValueType value_type;
    typedef ValueDistribution<ValueType> value_distribution_type;

    /**
     * Regular constructor
     */
    dense_transform_data_t (int N, int S, skylark::sketch::context_t& context)
        : N(N), S(S), context(context),
          distribution(),
          random_samples(context.allocate_random_samples_array(N * S, distribution)) {
        // No scaling in "raw" form
        scale = 1.0;
    }


    const dense_transform_data_t& get_data() const {
        return static_cast<const dense_transform_data_t&>(*this);
    }

protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    value_distribution_type distribution; /**< Distribution for samples */
    const skylark::utility::random_samples_array_t < value_distribution_type>
        random_samples;
    /**< Array of samples, to be lazily computed */
    double scale; /**< Scaling factor for the samples */
  };

} } /** namespace skylark::sketch */

#endif /** SKYLARK_DENSE_TRANSFORM_DATA_HPP */
