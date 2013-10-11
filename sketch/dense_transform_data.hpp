#ifndef DENSE_TRANSFORM_DATA_HPP
#define DENSE_TRANSFORM_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/randgen.hpp"

#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base class for all dense transforms. Essentially, it
 * holds the input and sketched matrix sizes and the array of samples
 * to be lazily computed.
 */
template <typename ValueType,
          template <typename> class ValueDistributionType>
struct dense_transform_data_t {
    typedef ValueType value_type;
    typedef ValueDistributionType<ValueType> value_distribution_type;

    /**
     * Regular constructor
     */
    dense_transform_data_t (int N, int S, skylark::sketch::context_t& context)
        : N(N), S(S), context(context),
          distribution(),
          random_samples(context.allocate_random_samples_array
              <value_type, value_distribution_type>
              (N * S, distribution)) {
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
  value_distribution_type distribution;
    const skylark::utility::random_samples_array_t
    <value_type, value_distribution_type> random_samples;
  double scale;
  };

} } /** namespace skylark::sketch */

#endif /** DENSE_TRANSFORM_DATA_HPP */
