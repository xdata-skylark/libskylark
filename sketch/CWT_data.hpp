#ifndef SKYLARK_CWT_DATA_HPP
#define SKYLARK_CWT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include "../utility/distributions.hpp"

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
struct CWT_data_t : public hash_transform_data_t<
    boost::random::uniform_int_distribution,
    utility::rademacher_distribution_t > {

    typedef hash_transform_data_t<
        boost::random::uniform_int_distribution,
        utility::rademacher_distribution_t > base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

    };

    CWT_data_t(int N, int S, base::context_t& context)
        : base_t(N, S, context, "CWT") {

        context = base_t::build();
    }

    CWT_data_t(int N, int S, const params_t& params, base::context_t& context)
        : base_t(N, S, context, "CWT") {

        context = base_t::build();
    }

    CWT_data_t(const boost::property_tree::ptree& pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "CWT") {
        base_t::build();
    }

    /**
     *  Serializes a sketch to a Boost property tree. This can be conveniently
     *  converted to other formats, e.g. to JSON and XML.
     *
     *  @return property_tree describing the sketch.
     */
    virtual boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const;

protected:

    CWT_data_t(int N, int S, const base::context_t& context, std::string type)
        : base_t(N, S, context, type) {

    }


};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CWT_DATA_HPP
