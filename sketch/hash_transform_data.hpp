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
  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef IdxDistributionType idx_distribution_t;
  typedef ValueDistributionType<ValueType> value_distribution_t;

  /**
   * Regular constructor
   */
  hash_transform_data_t (int N,
                         int S,
                         skylark::sketch::context_t& context)
  : N(N), S(S), context(context) {

    try {
      row_idx.resize(N);
      row_value.resize(N);
    } catch (std::bad_alloc ba) {
      SKYLARK_THROW_EXCEPTION (
        utility::allocation_exception()
        << utility::error_msg(ba.what()) );
    }

    
    skylark::utility::rng_array_t* idx_rng_ptr = 
        context.allocate_rng_array(N);
    skylark::utility::rng_array_t* value_rng_ptr = 
        context.allocate_rng_array(N);
    idx_distribution_t row_idx_distribution(0, S-1);
    value_distribution_t row_value_distribution;
   

    for (int i = 0; i < N; ++i) {
        skylark::utility::URNG_t idx_urng = (*idx_rng_ptr)[i];
        skylark::utility::URNG_t value_urng = (*value_rng_ptr)[i];
        row_idx[i] = row_idx_distribution(idx_urng);
        row_value[i] = row_value_distribution(value_urng);
    }
    delete idx_rng_ptr;
    delete value_rng_ptr;
  }

  hash_transform_data_t& get_data() {
    return static_cast<hash_transform_data_t&>(*this);
  }

  protected:

  const int N; /**< Input dimension  */
  const int S; /**< Output dimension  */
  skylark::sketch::context_t& context; /**< Context for this sketch */
  std::vector<int> row_idx; /**< precomputed row indices */
  std::vector<value_t> row_value; /**< precomputed scaling factors */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_HASH_TRANSFORM_DATA_HPP */
