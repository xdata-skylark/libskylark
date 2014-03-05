#ifndef SKYLARK_RFUT_DATA_HPP
#define SKYLARK_RFUT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {

/**
 * This is the base data class for RFUT transform. Essentially, it
 * holds the input and sketched matrix sizes and the random diagonal part.
 */
template <typename ValueType,
          typename ValueDistributionType>
struct RFUT_data_t {
    // Only for consistency reasons
    typedef ValueType value_type;
    typedef ValueDistributionType value_distribution_type;

    /**
     * Regular constructor
     */
    RFUT_data_t (int N, skylark::sketch::context_t& context)
        : _N(N), _context(context) {
        value_distribution_type distribution;
        D = context.generate_random_samples_array(N, distribution);
    }


    const RFUT_data_t& get_data() const {
        return static_cast<const RFUT_data_t&>(*this);
    }

    //TODO: inherit from (dense_)transform_t or serialize here
    //TODO: serialize distribution

protected:
    const int _N; /**< Input dimension  */
    skylark::sketch::context_t& _context; /**< Context for this sketch */
    std::vector<value_type> D; /**< Diagonal part */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFUT_DATA_HPP */
