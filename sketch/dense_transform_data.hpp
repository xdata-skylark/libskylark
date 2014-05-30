#ifndef SKYLARK_DENSE_TRANSFORM_DATA_HPP
#define SKYLARK_DENSE_TRANSFORM_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "../utility/randgen.hpp"

#include "sketch_transform_data.hpp"

#include "boost/smart_ptr.hpp"

namespace skylark { namespace sketch {

//FIXME: WHY DO WE NEED TO ALLOW COPY CONSTRUCTOR HERE (or more precisely in
//       dense_transform_Elemental)?
/**
 * This is the base data class for dense transforms. Essentially, it
 * holds the input and sketched matrix sizes and the array of samples
 * to be lazily computed.
 */
template <template <typename> class ValueDistribution>
struct dense_transform_data_t : public sketch_transform_data_t {
    typedef sketch_transform_data_t base_t;

    // Note: we always generate doubles for array values,
    // but when applying to floats the size can be reduced.
    typedef ValueDistribution<double> value_distribution_type;

    /**
     * Regular constructor
     */
    dense_transform_data_t (int N, int S, double scale, 
        base::context_t& context)
        : base_t(N, S, context, "DenseTransform"),
          scale(scale), distribution() {

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

    dense_transform_data_t(const dense_transform_data_t& other)
        : base_t(other), scale(other.scale), distribution(other.distribution),
          random_samples(other.random_samples)  {

    }

protected:

    dense_transform_data_t (int N, int S, double scale, 
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type),
          scale(scale),
          distribution() {

    }

    base::context_t build() {
        base::context_t ctx = base_t::build();
        random_samples = ctx.allocate_random_samples_array(_N * _S, distribution);
        return ctx;
    }

    double scale; /**< Scaling factor for the samples */
    value_distribution_type distribution; /**< Distribution for samples */
    skylark::utility::random_samples_array_t <value_distribution_type>
    random_samples;
    /**< Array of samples, to be lazily computed */


};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_DENSE_TRANSFORM_DATA_HPP */
