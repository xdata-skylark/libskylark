#ifndef RFT_DATA_HPP
#define RFT_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/randgen.hpp"

#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * This is the base class for all dense transforms. Essentially, it
 * holds the input and sketched matrix sizes and the array of samples
 * to be lazily computed.
 */
template <typename ValueType,
          template <typename> class UnderlyingValueDistributionType>
struct RFT_data_t {
    typedef ValueType value_type;
    typedef boost::random::uniform_real_distribution<>
    value_distribution_type;
    typedef skylark::sketch::dense_transform_data_t
    <value_type, UnderlyingValueDistributionType>
    underlying_data_type;

    /**
     * Regular constructor
     */
    RFT_data_t (int N, int S, double sigma, skylark::sketch::context_t& context)
        : N(N), S(S), sigma(sigma), context(context),
          underlying_data(N, S, context),
          shifts(S),
          scale(std::sqrt(2.0 / S)) {
        const double pi = boost::math::constants::pi<double>();
        value_distribution_type distribution(0, 2 * pi);
        skylark::utility::random_samples_array_t
            <value_type, value_distribution_type>
            random_samples =
            context.allocate_random_samples_array
            <value_type, value_distribution_type>
            (S, distribution);
        for (int i = 0; i < S; i++) {
            shifts[i] = random_samples[i];
        }
    }

    RFT_data_t& get_data() {
        return static_cast<RFT_data_t&>(*this);
    }

protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    double sigma; /**< Bandwidth (sigma)  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    underlying_data_type  underlying_data;
    std::vector<double> shifts;
    double scale;
};

} } /** namespace skylark::sketch */

#endif /** RFT_DATA_HPP */
