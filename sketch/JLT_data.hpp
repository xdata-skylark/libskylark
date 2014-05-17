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
                                   bstrand::normal_distribution > base_t;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    JLT_data_t(int N, int S, skylark::base::context_t& context)
        : base_t(N, S, context, "JLT") {
        base_t::scale = sqrt(1.0 / static_cast<double>(S));
        context = base_t::build();
    }

    JLT_data_t(boost::property_tree::ptree &json)
        : base_t(json, true) {
        base_t::scale = sqrt(1.0 / static_cast<double>(base_t::_S));
        base_t::build();
    }

protected:

    JLT_data_t(int N, int S, skylark::base::context_t& context, 
        std::string type)
        : base_t(N, S, context, type) {
        base_t::scale = sqrt(1.0 / static_cast<double>(S));
    }

    JLT_data_t(boost::property_tree::ptree &json, bool nobuild)
        : base_t(json, true) {
        base_t::scale = sqrt(1.0 / static_cast<double>(base_t::_S));
    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_DATA_HPP
