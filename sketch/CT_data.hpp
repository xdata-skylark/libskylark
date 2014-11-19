#ifndef SKYLARK_CT_DATA_HPP
#define SKYLARK_CT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <boost/random.hpp>
#include <boost/property_tree/ptree.hpp>

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Cauchy Transform (data)
 *
 * The CT is simply a dense random matrix with i.i.d Cauchy variables
 */
struct CT_data_t :
   public random_dense_transform_data_t<bstrand::cauchy_distribution> {

    typedef random_dense_transform_data_t<bstrand::cauchy_distribution> base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double C) : C(C) {

        }

        const double C;
    };

    CT_data_t(int N, int S, double C, skylark::base::context_t& context)
        : base_t(N, S, C / static_cast<double>(S), context, "CT"), _C(C) {

        context = base_t::build();
    }

    CT_data_t(int N, int S, const params_t& params,
        skylark::base::context_t& context)
        : base_t(N, S, params.C / static_cast<double>(S), context, "CT"),
          _C(params.C) {

        context = base_t::build();
    }

    CT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            pt.get<double>("C") / pt.get<double>("S"),
            base::context_t(pt.get_child("creation_context")), "CT"),
        _C(pt.get<double>("C")) {

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
        : base_t(N, S, C / static_cast<double>(S), context, type), _C(C) {

    }

private:

    double _C;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CT_DATA_HPP
