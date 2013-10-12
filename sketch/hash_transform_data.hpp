#ifndef SKYLARK_HASH_TRANSFORM_DATA_HPP
#define SKYLARK_HASH_TRANSFORM_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/randgen.hpp"

#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base class for all the hashing transforms. Essentially, it
 * holds on to a context, and to some random numbers that it has generated
 * both for the scaling factor and for the row/col indices.
 */
template <typename IndexType,
          typename ValueType,
          typename IdxDistributionType,
          template <typename> class ValueDistributionType>
struct hash_transform_data_t {
  typedef IndexType index_type;
  typedef ValueType value_type;
  typedef IdxDistributionType idx_distribution_type;
  typedef ValueDistributionType<ValueType> value_distribution_type;

  /**
   * Regular constructor
   */
  hash_transform_data_t (int N,
                         int S,
                         skylark::sketch::context_t& context)
  : N(N), S(S), context(context) {

    idx_distribution_type row_idx_distribution(0, S-1);
    value_distribution_type row_value_distribution;

    row_idx = context.generate_random_samples_array
        <int, idx_distribution_type>
        (N, row_idx_distribution);
    row_value = context.generate_random_samples_array
        <value_type, value_distribution_type>
        (N, row_value_distribution);
  }

  const hash_transform_data_t& get_data() const {
    return static_cast<const hash_transform_data_t&>(*this);
  }

  protected:

  const int N; /**< Input dimension  */
  const int S; /**< Output dimension  */
  skylark::sketch::context_t& context; /**< Context for this sketch */
  std::vector<int> row_idx; /**< precomputed row indices */
  std::vector<value_type> row_value; /**< precomputed scaling factors */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_HASH_TRANSFORM_DATA_HPP */
