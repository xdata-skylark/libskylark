#ifndef SKYLARK_CT_DATA_HPP
#define SKYLARK_CT_DATA_HPP

#include <boost/random.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "transform_data.hpp"
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
                                   bstrand::cauchy_distribution > base_t;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    CT_data_t(int N, int S, double C, skylark::base::context_t& context)
        : base_t(N, S, context, "CT"), _C(C) {
        base_t::scale = C / static_cast<double>(S);
    }

    CT_data_t(const boost::property_tree::ptree &sketch,
              skylark::base::context_t& context)
        : base_t(sketch, context),
        _C(sketch.get<double>("sketch.c")) {

        base_t::scale = _C / static_cast<double>(base_t::_S);
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk, const CT_data_t<ValueT> &data);
private:

    double _C;
};

template <typename ValueType>
boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const CT_data_t<ValueType> &data) {

    sk << static_cast<const transform_data_t&>(data);
    sk.put("sketch.c", data._C);
    return sk;
}

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_DATA_HPP
