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
template <typename ValueType,
          template <typename> class ValueDistribution>
struct dense_transform_data_t : public sketch_transform_data_t {
    typedef sketch_transform_data_t base_t;

    // For reasons of naming consistency
    typedef ValueType value_type;
    typedef ValueDistribution<ValueType> value_distribution_type;

    /**
     * Regular constructor
     */
    dense_transform_data_t (int N, int S, base::context_t& context,
                            std::string type = "")
        : base_t(N, S, context, type),
          distribution() {

        // No scaling in "raw" form
        scale = 1.0;
        context = build();
    }

    dense_transform_data_t (const boost::property_tree::ptree& json)
        : base_t(json, true),
          distribution() {

        // No scaling in "raw" form
        scale = 1.0;
        build();
    }

protected:

    dense_transform_data_t (int N, int S, base::context_t& context,
        std::string type, bool nobuild)
        : base_t(N, S, context, type),
          distribution() {

        // No scaling in "raw" form
        scale = 1.0;
    }

    dense_transform_data_t (const boost::property_tree::ptree& json,
        bool nobuild)
        : base_t(json),
          distribution() {

        // No scaling in "raw" form
        scale = 1.0;
    }

    base::context_t build() {
        base::context_t tmp = base_t::build();
        random_samples = tmp.allocate_random_samples_array(_N * _S, distribution);
        return tmp;
    }
    value_distribution_type distribution; /**< Distribution for samples */
    boost::shared_ptr<
        skylark::utility::random_samples_array_t <value_distribution_type> >
            random_samples;
    /**< Array of samples, to be lazily computed */
    double scale; /**< Scaling factor for the samples */

};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_DENSE_TRANSFORM_DATA_HPP */
