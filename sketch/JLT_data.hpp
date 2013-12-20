#ifndef SKYLARK_JLT_DATA_HPP
#define SKYLARK_JLT_DATA_HPP

#include "dense_transform_data.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Johnson-Lindenstrauss Transform (data).
 *
 * The JLT is simply a dense random matrix with i.i.d normal entries.
 */
template < typename ValueType>
struct JLT_data_t :
   public dense_transform_data_t<ValueType,
                                         bstrand::normal_distribution > {

    typedef dense_transform_data_t<ValueType,
                               bstrand::normal_distribution > Base;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    JLT_data_t(int N, int S, skylark::sketch::context_t& context)
        : Base(N, S, context) {
        Base::scale = sqrt(1.0 / static_cast<double>(S));
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_DATA_HPP
