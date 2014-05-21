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

    JLT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "JLT") {
        base_t::scale = sqrt(1.0 / static_cast<double>(base_t::_S));
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
        // TODO: serialize value_type?
        return pt;
    }

protected:

    JLT_data_t(int N, int S, const skylark::base::context_t& context,
        std::string type)
        : base_t(N, S, context, type) {
        base_t::scale = sqrt(1.0 / static_cast<double>(S));
    }


};

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_DATA_HPP
