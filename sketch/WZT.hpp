#ifndef SKYLARK_WZT_HPP
#define SKYLARK_WZT_HPP

#include <boost/random.hpp>
#include "hash_transform.hpp"

namespace skylark { namespace sketch {

typedef boost::random::uniform_int_distribution<int> uniform_t;

/**
 * Woodruff-Zhang Transform
 *
 * Hash trasfrom with reciprocal exponential variables.
 * TODO current implementation is only one sketch index, when for 1 <= p <= 2
 *      you want more than one.
 */
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct WZT_t : public hash_transform_t<
    InputMatrixType, OutputMatrixType,
    boost::random::uniform_int_distribution,
    boost::random::exponential_distribution >
{
    typedef hash_transform_t<InputMatrixType, OutputMatrixType,
                             boost::random::uniform_int_distribution,
                             boost::random::exponential_distribution> Base;
    WZT_t(int N, int S, double p, context_t& context)
        : Base(N, S, context) {

        // Since the distribution depends on the target p we have to pass p as
        // a parameter. We also cannot just use the distribution as template.
        // The only solution I found is to let the base class generate the
        // numbers and then modify them to the correct distribution.
        for(int i = 0; i < N; i++)
             Base::row_value[i] = pow(1.0 / Base::row_value[i], 1.0 / p);
     }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_WZT_HPP
