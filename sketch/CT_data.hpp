#ifndef SKYLARK_CT_DATA_HPP
#define SKYLARK_CT_DATA_HPP

#include "dense_transform_data.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Cauchy Transform (data)
 *
 * The CT is simply a dense random matrix with i.i.d Cauchy variables
 */
template < typename ValueType>
struct CT_data_t :
   public dense_transform_data_t<ValueType,
                                 bstrand::cauchy_distribution > {

    typedef dense_transform_data_t<ValueType,
                                   bstrand::cauchy_distribution > Base;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    CT_data_t(int N, int S, double C, skylark::sketch::context_t& context)
        : Base(N, S, context) {
        Base::scale = C / static_cast<double>(S);
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_DATA_HPP
