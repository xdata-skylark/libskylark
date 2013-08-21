#ifndef CWT_HPP
#define CWT_HPP

#include "../utility/distributions.hpp"
#include "hash_transform.hpp"

namespace skylark { namespace sketch { 
typedef boost::random::uniform_int_distribution<int> uniform_t;

// CW'12 and MM'13 L2 transformation (= OSNAP with s = 1)
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct CWT_t : public hash_transform_t<
    InputMatrixType, OutputMatrixType,
    uniform_t, utility::rademacher_distribution_t > {
    CWT_t(int N, int S, context_t& context)
        : hash_transform_t<InputMatrixType, OutputMatrixType,
          uniform_t, utility::rademacher_distribution_t>(N, S, context)
    {}

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    CWT_t(CWT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
        : hash_transform_t<InputMatrixType, OutputMatrixType,
          uniform_t, utility::rademacher_distribution_t>(other)
    {}

};

} } // namespace sketch

#endif // CWT_HPP
