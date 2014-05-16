#ifndef SKYLARK_CWT_DATA_HPP
#define SKYLARK_CWT_DATA_HPP

#include "../utility/distributions.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

/**
 * Clarkson-Woodruff Transform (data)
 *
 * Clarkson-Woodruff Transform is essentially the CountSketch
 * sketching originally suggested by Charikar et al.
 * Analysis by Clarkson and Woodruff in STOC 2013 shows that
 * this is sketching scheme can be used to build a subspace embedding.
 *
 * CWT was additionally analyzed by Meng and Mahoney (STOC'13) and is equivalent
 * to OSNAP with s=1.
 */
template<typename IndexType, typename ValueType>
struct CWT_data_t : public hash_transform_data_t<
    IndexType, ValueType,
    boost::random::uniform_int_distribution,
    utility::rademacher_distribution_t > {

    typedef hash_transform_data_t<
        IndexType, ValueType,
        boost::random::uniform_int_distribution,
        utility::rademacher_distribution_t > base_t;

    CWT_data_t(int N, int S, base::context_t& context)
        : base_t(N, S, context, "CWT", true) {

        context = base_t::build();
    }

    CWT_data_t(const boost::property_tree::ptree& json)
        : base_t(json, true) {

        base_t::build();
    }

protected:

    CWT_data_t(int N, int S, base::context_t& context, bool nobuild)
        : base_t(N, S, context, "CWT", true) {

    }

    CWT_data_t(const boost::property_tree::ptree& json, bool nobuild)
        : base_t(json, true) {

    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CWT_DATA_HPP
