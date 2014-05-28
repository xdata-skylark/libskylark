#ifndef SKYLARK_CT_DATA_HPP
#define SKYLARK_CT_DATA_HPP

#include <boost/random.hpp>
#include <boost/property_tree/ptree.hpp>

#include "sketch_transform_data.hpp"
#include "dense_transform_data.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Cauchy Transform (data)
 *
 * The CT is simply a dense random matrix with i.i.d Cauchy variables
 */
struct CT_data_t :
   public dense_transform_data_t<bstrand::cauchy_distribution> {

    typedef dense_transform_data_t<bstrand::cauchy_distribution> base_t;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    CT_data_t(int N, int S, double C, skylark::base::context_t& context)
        : base_t(N, S, context, "CT"), _C(C) {
        base_t::scale = C / static_cast<double>(S);
        context = base_t::build();
    }

    CT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "CT"),
        _C(pt.get<double>("C")) {

        base_t::scale = _C / static_cast<double>(base_t::_S);
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
        pt.put("C", _C);
        return pt;
    }

protected:

    CT_data_t(int N, int S, double C, const skylark::base::context_t& context, 
        std::string type)
        : base_t(N, S, context, type), _C(C) {

        base_t::scale = C / static_cast<double>(S);
    }

private:

    double _C;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CT_DATA_HPP
