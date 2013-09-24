/**
 * Some additional useful distributions (only bare operator()).
 */
#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include <boost/random.hpp>

namespace skylark {
namespace utility {

/**
 * Radamachar distribution - +1 and -1 with equal probability.
 */
template< typename ValueType >
struct rademacher_distribution_t {

    template< typename URNG >
    ValueType operator()(URNG &prng) const {
        double probabilities[] = { 0.5, 0.0, 0.5 };
        boost::random::discrete_distribution<> dist(probabilities);
        return static_cast<ValueType>(dist(prng)) - 1.0;
    }

    void reset() {}

};

} // namespace utility
} // namespace skylark

#endif // DISTRIBUTIONS_HPP
