#ifndef SKYLARK_MMT_DATA_HPP
#define SKYLARK_MMT_DATA_HPP

#include "../utility/distributions.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

/**
 * Meng-Mahoney Transform (data)
 *
 * Meng-Mahoney Transform is very similar to the Clarkson-Woodruff Transform:
 * it replaces the +1/-1 diagonal with Cauchy random enteries. Thus, it
 * provides a low-distortion of l1-norm subspace embedding.
 *
 * See Meng and Mahoney's STOC'13 paper.
 */
template<typename IndexType, typename ValueType>
struct MMT_data_t : public hash_transform_data_t<
    IndexType, ValueType,
    boost::random::uniform_int_distribution,
    boost::random::cauchy_distribution > {

    typedef hash_transform_data_t<
        IndexType, ValueType,
        boost::random::uniform_int_distribution,
        boost::random::cauchy_distribution > base_t;

    MMT_data_t(int N, int S, skylark::base::context_t& context)
        : base_t(N, S, context, "MMT") {

        context = base_t::build();
   }

   MMT_data_t(const boost::property_tree::ptree &json)
       : base_t(json, true) {

        base_t::build();
   }

protected:

    MMT_data_t(int N, int S, skylark::base::context_t& context, 
        std::string type)
        : base_t(N, S, context, type) {

   }

    MMT_data_t(const boost::property_tree::ptree &json, bool nobuild)
       : base_t(json, true) {

   }


};

} } /** namespace skylark::sketch */

#endif // SKYLARK_MMT_DATA_HPP
