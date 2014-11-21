#ifndef SKYLARK_RLT_DATA_HPP
#define SKYLARK_RLT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {


/**
 * Random Laplace Transform (data)
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a semigroup kernel.
 *
 * See:
 *
 * Jiyan Yang, Vikas Sindhwani, Quanfu Fan, Haim Avron, Michael Mahoney
 * Random Laplace Feature Maps for Semigroup Kernels on Histograms
 * CVPR 2014
 *
 */
template <template <typename> class KernelDistribution>
struct RLT_data_t : public sketch_transform_data_t {

    typedef double value_type;
    typedef random_dense_transform_data_t<KernelDistribution>
    underlying_data_type;
    typedef sketch_transform_data_t base_t;

    RLT_data_t (int N, int S, double inscale, double outscale,
        base::context_t& context)
        : base_t(N, S, context, "RLT"), _inscale(inscale),
          _outscale(outscale) {

        context = build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual
    boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic RLT transform"));

        return boost::property_tree::ptree();
    }

protected:

    typedef typename underlying_data_type::value_accessor_type accessor_type;


    RLT_data_t (int N, int S, double inscale, double outscale,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type), _inscale(inscale),
          _outscale(outscale) {

    }

   base::context_t build() {
       base::context_t ctx = base_t::build();
       _underlying_data = boost::shared_ptr<underlying_data_type>(new
           underlying_data_type(base_t::_N, base_t::_S, _inscale, ctx));
       return ctx;
   }

    double _inscale; 
    double _outscale; /** Scaling for exponential factor */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */

};

/**
 * Random Features for Exponential Semigroup
 */
struct ExpSemigroupRLT_data_t :
        public RLT_data_t<utility::standard_levy_distribution_t> {

    typedef RLT_data_t<utility::standard_levy_distribution_t > base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double beta) : beta(beta) {

        }

        const double beta;
    };

    ExpSemigroupRLT_data_t(int N, int S, double beta,
        base::context_t& context)
        : base_t(N, S, beta * beta / 2, std::sqrt(1.0 / S), 
            context, "ExpSemigroupRLT"), _beta(beta) {

        context = base_t::build();
    }

    ExpSemigroupRLT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, params.beta * params.beta / 2, std::sqrt(1.0 / S), 
            context, "ExpSemigroupRLT"), _beta(params.beta) {

        context = base_t::build();
    }

    ExpSemigroupRLT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            pt.get<double>("beta") * pt.get<double>("beta") / 2,
            std::sqrt(1.0 / pt.get<double>("S")),
            base::context_t(pt.get_child("creation_context")), "ExpSemiGroupRLT"),
        _beta(pt.get<double>("beta")) {

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
        pt.put("beta", _beta);
        return pt;
    }

protected:
    ExpSemigroupRLT_data_t(int N, int S, double beta,
        const base::context_t& context, std::string type)
        : base_t(N, S,  beta * beta / 2, std::sqrt(1.0 / S), context, type), 
          _beta(beta) {

    }


private:
    const double _beta;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RLT_DATA_HPP */
