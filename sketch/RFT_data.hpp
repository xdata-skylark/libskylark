#ifndef SKYLARK_RFT_DATA_HPP
#define SKYLARK_RFT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "dense_transform_data.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {

/**
 * This is the base data class for RFT. Essentially, it
 * holds the input and sketched matrix sizes, the vector of shifts
 * and the data of the underlying transform.
 */
template <typename ValueType,
          template <typename> class UnderlyingValueDistribution>
struct RFT_data_t {
    // Typedef value, distribution and data types so that we can use them
    // regularly and consistently
    typedef ValueType value_type;
    typedef boost::random::uniform_real_distribution<> value_distribution_type;
    typedef skylark::sketch::dense_transform_data_t<value_type,
                                                    UnderlyingValueDistribution>
    underlying_data_type;

    /**
     * Regular constructor
     */
    RFT_data_t (int N, int S, double sigma, skylark::sketch::context_t& context)
        : N(N), S(S), sigma(sigma), context(context),
          underlying_data(N, S, context),
          scale(std::sqrt(2.0 / S)) {
        const double pi = boost::math::constants::pi<double>();
        value_distribution_type distribution(0, 2 * pi);
        shifts = context.generate_random_samples_array<double,
                                                       value_distribution_type>
            (S, distribution);
    }


    const RFT_data_t& get_data() const {
        return static_cast<const RFT_data_t&>(*this);
    }


protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    const double sigma; /**< Bandwidth (sigma)  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    const underlying_data_type underlying_data;
    /**< Data of the underlying dense transformation */
    const double scale; /** Scaling for trigonometric factor */
    std::vector<double> shifts; /** Shifts for scaled trigonometric factor */
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFT_DATA_HPP */
