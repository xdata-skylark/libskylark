#ifndef SKYLARK_HASH_TRANSFORM_DATA_HPP
#define SKYLARK_HASH_TRANSFORM_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#else
#include "sketch_transform_data.hpp"
#endif

#include <vector>

namespace skylark { namespace sketch {

/**
 * This is the base data class for all the hashing transforms. Essentially, it
 * holds on to a context, and to some random numbers that it has generated
 * both for the scaling factor and for the row/col indices.
 */
template <template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_data_t : public sketch_transform_data_t {
    typedef sketch_transform_data_t base_t;

    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<double> value_distribution_type;

    /**
     *  Constructs the data for a hashing sketch.
     *  @param N
     *  @param S
     *  @param context
     */
    hash_transform_data_t (int N, int S, base::context_t& context)
        : base_t(N, S, context, "HashTransform") {
        context = build();
    }

    virtual boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic hash transform"));

        return boost::property_tree::ptree();
    }

    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Trying to create concrete transform of hash_transform_data_t"));

        return nullptr;
    }

protected:

    hash_transform_data_t (int N, int S, const base::context_t& context,
        const std::string type)
        : base_t(N, S, context, type) {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();

        idx_distribution_type row_idx_distribution(0, _S - 1);
        value_distribution_type row_value_distribution;

        row_idx   = ctx.generate_random_samples_array(
                        _N, row_idx_distribution);
        row_value = ctx.generate_random_samples_array(
                        _N, row_value_distribution);

        return ctx;
    }

    std::vector<size_t> row_idx; /**< precomputed row indices */
    std::vector<double> row_value; /**< precomputed scaling factors */

    inline void finalPos(size_t &rowid, size_t &colid, columnwise_tag) const {
        rowid = row_idx[rowid];
    }

    inline void finalPos(size_t &rowid, size_t &colid, rowwise_tag) const {
        colid = row_idx[colid];
    }

    inline double getValue(size_t rowid, size_t colid, columnwise_tag) const {
        return row_value[rowid];
    }

    inline double getValue(size_t rowid, size_t colid, rowwise_tag) const {
        return row_value[colid];
    }

    inline void get_res_size(int &rows, int &cols, columnwise_tag) const {
        rows = _S;
    }

    inline void get_res_size(int &rows, int &cols, rowwise_tag) const {
        cols = _S;
    }
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_HASH_TRANSFORM_DATA_HPP */
