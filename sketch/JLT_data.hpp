#ifndef SKYLARK_JLT_DATA_HPP
#define SKYLARK_JLT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Johnson-Lindenstrauss Transform (data).
 *
 * The JLT is simply a dense random matrix with i.i.d normal entries.
 */
struct JLT_data_t :
   public random_dense_transform_data_t<bstrand::normal_distribution> {

    typedef random_dense_transform_data_t<bstrand::normal_distribution> base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

    };

    JLT_data_t(int N, int S, skylark::base::context_t& context)
        : base_t(N, S, sqrt(1.0 / static_cast<double>(S)),
            bstrand::normal_distribution<double>(), context, "JLT") {

        context = base_t::build();
    }

    JLT_data_t(int N, int S, const params_t& params,
        skylark::base::context_t& context)
        : base_t(N, S, sqrt(1.0 / static_cast<double>(S)),
            bstrand::normal_distribution<double>(), context, "JLT") {

        context = base_t::build();
    }

    JLT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            sqrt(1.0 / pt.get<double>("S")),
            bstrand::normal_distribution<double>(),
            base::context_t(pt.get_child("creation_context")), "JLT") {

        base_t::build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
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

    JLT_data_t(int N, int S, const skylark::base::context_t& context,
        std::string type)
        : base_t(N, S, sqrt(1.0 / static_cast<double>(S)), 
            bstrand::normal_distribution<double>(),
            context, type) {

    }


};

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_DATA_HPP
