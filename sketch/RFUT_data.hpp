#ifndef SKYLARK_RFUT_DATA_HPP
#define SKYLARK_RFUT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"

#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base data class for RFUT transform. Essentially, it
 * holds the input and sketched matrix sizes and the random diagonal part.
 */
template <typename ValueDistributionType>
struct RFUT_data_t {
    // Only for consistency reasons
    typedef ValueDistributionType value_distribution_type;

    /**
     * Regular constructor
     */
    RFUT_data_t (int N, skylark::base::context_t& context)
        : _N(N), _creation_context(context) {

        build();
    }

    // TODO support to_ptree (challenging part: the distribution).

protected:
    int _N; /**< Input dimension  */

    /// Store the context on creation for serialization
    const base::context_t _creation_context;

    std::vector<double> D; /**< Diagonal part */

    base::context_t build() {
        base::context_t ctx = _creation_context;

        value_distribution_type distribution;
        D = ctx.generate_random_samples_array(_N, distribution);

        return ctx;
    }
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFUT_DATA_HPP */
