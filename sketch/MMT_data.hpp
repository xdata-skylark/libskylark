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

    MMT_data_t(const boost::property_tree::ptree& pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "MMT") { 
        base_t::build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual
    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        // TODO: serialize index_type and value_type?
        return pt;
    }

protected:

    MMT_data_t(int N, int S, const skylark::base::context_t& context, 
        std::string type)
        : base_t(N, S, context, type) {

   }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_MMT_DATA_HPP
