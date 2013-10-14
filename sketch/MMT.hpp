#ifndef SKYLARK_MMT_HPP
#define SKYLARK_MMT_HPP

#include <boost/random.hpp>
#include "hash_transform.hpp"

namespace skylark { namespace sketch {

// L1 transformation described in MM'13 (= SW'10)
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct MMT_t : public hash_transform_t<
    InputMatrixType, OutputMatrixType,
    uniform_t, boost::random::cauchy_distribution > {
    MMT_t(int N, int S, context_t& context)
        : hash_transform_t<InputMatrixType, OutputMatrixType,
          uniform_t, boost::random::cauchy_distribution>(N, S, context)
    {}
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_MMT_HPP
