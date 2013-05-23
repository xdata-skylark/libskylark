#ifndef MMT_HPP
#define MMT_HPP

#include "config.h"

#include "utility/distributions.hpp"
#include "sparset.hpp"

namespace skylark {
namespace sketch {

// L1 transformation described in MM'13 (= SW'10)
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct MMT_t : public hash_transform_t<
    InputMatrixType, OutputMatrixType,
    uniform_t, boost::random::cauchy_distribution >
{
    MMT_t(int N, int S, context_t& context)
        : hash_transform_t<InputMatrixType, OutputMatrixType,
          uniform_t, boost::random::cauchy_distribution>(N, S, context)
    {}
};


} // namespace sketch
} // namespace skylark

#endif // MMT_HPP
