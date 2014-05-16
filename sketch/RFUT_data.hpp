#ifndef SKYLARK_RFUT_DATA_HPP
#define SKYLARK_RFUT_DATA_HPP

#include <vector>

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "../base/context.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {

/**
 * This is the base data class for RFUT transform. Essentially, it
 * holds the input and sketched matrix sizes and the random diagonal part.
 */
template <typename ValueType,
          typename ValueDistributionType>
struct RFUT_data_t {
    // Only for consistency reasons
    typedef ValueType value_type;
    typedef ValueDistributionType value_distribution_type;

    /**
     * Regular constructor
     */
    RFUT_data_t (int N, skylark::base::context_t& context, bool init = true)
        : _N(N), _creation_context(context) {

        if(init) build();
    }


    //TODO: inherit from (dense_)transform_t or serialize here
    //TODO: serialize distribution
    RFUT_data_t (boost::property_tree::ptree &json, bool init = true)
        : _creation_context(json) {

        std::vector<int> dims;
        BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                      json.get_child("sketch.size")) {

            std::istringstream i(v.second.data());
            int x;
            if (!(i >> x)) dims.push_back(0);
            dims.push_back(x);
        }

        _N = dims[0];

        //_type = json.get<std::string>("sketch.type");
    }

    template< typename VT, typename VDT >
    friend std::istream& operator>>(std::istream &in,
            RFUT_data_t<VT, VDT> &data);

    /**
     *  Serializes a sketch to a string.
     *  @param[out] dump containing serialized JSON object
     */
    template< typename VT, typename VDT >
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const RFUT_data_t<VT, VDT> &data);

protected:
    int _N; /**< Input dimension  */

    /// Store the context on creation for serialization
    const base::context_t _creation_context;

    std::vector<value_type> D; /**< Diagonal part */


    base::context_t build() {
        base::context_t ctx = _creation_context;

        value_distribution_type distribution;
        D = ctx.generate_random_samples_array(_N, distribution);

        return ctx;
    }
};


#if 0
/// serialize the sketch
template <typename ValueType,
          typename ValueDistributionType>
boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const RFUT_data_t<ValueType, ValueDistributionType> &data) {

    sk.put("sketch.type", "RFUT");
    sk.put("sketch.version", "0.1");

    boost::property_tree::ptree size;
    boost::property_tree::ptree size_n, size_s;
    size_n.put("", data._N);
    size.push_back(std::make_pair("", size_n));
    sk.add_child("sketch.size", size);

    sk << data._creation_context;

    return sk;
}
#endif

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFUT_DATA_HPP */
