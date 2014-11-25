#ifndef SKYLARK_RANDOM_DENSE_TRANSFORM_DATA_HPP
#define SKYLARK_RANDOM_DENSE_TRANSFORM_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/randgen.hpp"
#include "boost/smart_ptr.hpp"

namespace skylark { namespace sketch {

template <template <typename> class ValueDistribution>
struct random_dense_transform_data_t :
        public dense_transform_data_t<
    utility::random_samples_array_t< ValueDistribution<double> > > {

    // Note: we always generate doubles for array values,
    // but when applying to floats the size can be reduced.
    typedef double value_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef utility::random_samples_array_t<value_distribution_type>
        value_accessor_type;

    typedef dense_transform_data_t<value_accessor_type> base_t;

    /**
     * Regular constructor
     */
    random_dense_transform_data_t (int N, int S, double scale,
        base::context_t& context)
        : base_t(N, S, scale, context, "DistributionDenseTransform"),
          distribution() {

        // No scaling in "raw" form
        context = build();
    }

    virtual
    boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic dense transform"));

        return boost::property_tree::ptree();
    }

    random_dense_transform_data_t(const random_dense_transform_data_t& other)
        : base_t(other), distribution(other.distribution) {

    }


protected:

    random_dense_transform_data_t (int N, int S, double scale,
        const base::context_t& context, std::string type)
        : base_t(N, S, scale, context, type),
          distribution() {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();
        size_t array_size = static_cast<size_t>(base_t::_N) *
            static_cast<size_t>(base_t::_S);
        base_t::entries =
            ctx.allocate_random_samples_array(array_size, distribution);
        return ctx;
    }

    value_distribution_type distribution; /**< Distribution for samples */


};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RANDOM_DENSE_TRANSFORM_DATA_HPP */
