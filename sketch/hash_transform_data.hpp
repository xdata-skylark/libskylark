#ifndef SKYLARK_HASH_TRANSFORM_DATA_HPP
#define SKYLARK_HASH_TRANSFORM_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/randgen.hpp"
#include "../utility/exception.hpp"
#include "../utility/simple_json_parser.hpp"

#include "transform_data.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for all the hashing transforms. Essentially, it
 * holds on to a context, and to some random numbers that it has generated
 * both for the scaling factor and for the row/col indices.
 */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_data_t : public transform_data_t {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef IdxDistributionType<IndexType> idx_distribution_type;
    typedef ValueDistribution<ValueType> value_distribution_type;

    /**
     *  Constructs the data for a hashing sketch.
     *  @param N
     *  @param S
     *  @param context
     */
    hash_transform_data_t (int N, int S, skylark::sketch::context_t& context,
                           const std::string name = "")
        : transform_data_t(N, S, context, name) {

        _populate();
    }

    /**
     *  Load a serialized sketch from a file.
     *  @param[in] json_filename
     *  @param[in] context
     */
    hash_transform_data_t (const std::string json_filename,
                           context_t& context)
        : transform_data_t(json_filename, context) {

        _populate();
    }

    const hash_transform_data_t& get_data() const {
        return static_cast<const hash_transform_data_t&>(*this);
    }

protected:
    std::vector<index_type> row_idx; /**< precomputed row indices */
    std::vector<value_type> row_value; /**< precomputed scaling factors */

private:

    void _populate() {
        idx_distribution_type row_idx_distribution(0, S-1);
        value_distribution_type row_value_distribution;

        row_idx   = context.generate_random_samples_array(
                        N, row_idx_distribution);
        row_value = context.generate_random_samples_array(
                        N, row_value_distribution);
    }
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_HASH_TRANSFORM_DATA_HPP */
