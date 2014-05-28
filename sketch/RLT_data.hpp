#ifndef SKYLARK_RLT_DATA_HPP
#define SKYLARK_RLT_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "sketch_transform_data.hpp"
#include "dense_transform_data.hpp"
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
 * Random Laplace Feature Maps for Semigroup Kernels on Histograms
 *
 */
template <template <typename> class KernelDistribution>
struct RLT_data_t : public sketch_transform_data_t {

    typedef dense_transform_data_t<KernelDistribution> underlying_data_type;
    typedef sketch_transform_data_t base_t;

    /**
     * Regular constructor
     */
    RLT_data_t (int N, int S, skylark::base::context_t& context)
        : base_t(N, S, context, "RLT"), _val_scale(1),
          _scale(std::sqrt(1.0 / base_t::_S)) {
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
    RLT_data_t (int N, int S, const skylark::base::context_t& context,
        std::string type)
        : base_t(N, S, context, type), _val_scale(1),
          _scale(std::sqrt(1.0 / base_t::_S)) {

    }

   base::context_t build() {
       base::context_t ctx = base_t::build();
       _underlying_data = boost::shared_ptr<underlying_data_type>(new
           underlying_data_type(base_t::_N, base_t::_S, ctx));
       return ctx;
   }

    double _val_scale; /**< Bandwidth (sigma)  */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */
    const double _scale; /** Scaling for trigonometric factor */
};

/**
 * Random Features for Exponential Semigroup
 */
struct ExpSemigroupRLT_data_t :
        public RLT_data_t<utility::standard_levy_distribution_t> {

    typedef RLT_data_t<utility::standard_levy_distribution_t > base_t;

    /**
     * Constructor
     */
    ExpSemigroupRLT_data_t(int N, int S, double beta,
        skylark::base::context_t& context)
        : base_t(N, S, context, "ExpSemigroupRLT"), _beta(beta) {

        base_t::_val_scale = beta * beta / 2;
        context = base_t::build();
    }

    ExpSemigroupRLT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            base::context_t(pt.get_child("creation_context")), "ExpSemiGroupRLT"),
        _beta(pt.get<double>("beta")) {

        base_t::_val_scale = _beta * _beta / 2;
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
        const skylark::base::context_t& context, std::string type)
        : base_t(N, S, context, type), _beta(beta) {

        base_t::_val_scale = beta * beta / 2;
    }


private:
    const double _beta;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RLT_DATA_HPP */
