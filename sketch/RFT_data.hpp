#ifndef SKYLARK_RFT_DATA_HPP
#define SKYLARK_RFT_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "sketch_transform_data.hpp"
#include "dense_transform_data.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {


/**
 * Random Fourier Transform (data)
 *
 * Sketch transform into Euclidean space of functions in an RKHS
 * implicitly defined by a vector and a shift-invariant kernel.
 *
 * See:
 * Ali Rahimi and Benjamin Recht
 * Random Features for Large-Scale Kernel Machines
 * NIPS 2007.
 */
template <template <typename> class KernelDistribution>
struct RFT_data_t : public sketch_transform_data_t {

    typedef dense_transform_data_t<KernelDistribution> underlying_data_type;
    typedef sketch_transform_data_t base_t;

    RFT_data_t (int N, int S, double inscale, double outscale,
        base::context_t& context)
        : base_t(N, S, context, "RFT"), _inscale(inscale),
          _outscale(outscale) {

        context = build();
    }

    virtual
    boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic RFT transform"));

        return boost::property_tree::ptree();
    }

protected:
    RFT_data_t (int N, int S, double inscale, double outscale,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type),  _inscale(inscale),
          _outscale(outscale) {

    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        _underlying_data = boost::shared_ptr<underlying_data_type>(new
            underlying_data_type(base_t::_N, base_t::_S, _inscale, ctx));

        const double pi = boost::math::constants::pi<double>();
        boost::random::uniform_real_distribution<double>
            distribution(0, 2 * pi);
        _shifts = ctx.generate_random_samples_array(base_t::_S, distribution);
        return ctx;
    }

    double _inscale;
    double _outscale; /** Scaling for trigonometric factor */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */
    std::vector<double> _shifts; /** Shifts for scaled trigonometric factor */
};

struct GaussianRFT_data_t :
        public RFT_data_t<bstrand::normal_distribution> {

    typedef RFT_data_t<bstrand::normal_distribution > base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma) : sigma(sigma) {

        }

        const double sigma;
    };

    GaussianRFT_data_t(int N, int S, double sigma,
        base::context_t& context)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, "GaussianRFT"), 
          _sigma(sigma) {

        context = base_t::build();
    }

    GaussianRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S), context, "GaussianRFT"), 
          _sigma(params.sigma) {

        context = base_t::build();
    }

    GaussianRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("sigma"),
            std::sqrt(2.0 / pt.get<double>("S")),
            base::context_t(pt.get_child("creation_context")), "GaussianRFT"),
        _sigma(pt.get<double>("sigma")) {

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
        pt.put("sigma", _sigma);
        return pt;
    }

protected:
    GaussianRFT_data_t(int N, int S, double sigma,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, type),
          _sigma(sigma) {

    }

private:
    const double _sigma;
};

struct LaplacianRFT_data_t :
        public RFT_data_t<bstrand::cauchy_distribution> {

    typedef RFT_data_t<bstrand::cauchy_distribution > base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma) : sigma(sigma) {

        }

        const double sigma;
    };

    LaplacianRFT_data_t(int N, int S, double sigma,
        base::context_t& context)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, "LaplacianRFT"),
        _sigma(sigma) {

        context = base_t::build();
    }

    LaplacianRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S), context, "LaplacianRFT"),
        _sigma(params.sigma) {

        context = base_t::build();
    }

    LaplacianRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("sigma"),
            std::sqrt(2.0 / pt.get<double>("S")),
            base::context_t(pt.get_child("creation_context")), "LaplacianRFT"),
        _sigma(pt.get<double>("sigma")) {

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
        pt.put("sigma", _sigma);
        return pt;
    }

protected:

    LaplacianRFT_data_t(int N, int S, double sigma,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, type), 
          _sigma(sigma) {

    }

private:
    const double _sigma;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFT_DATA_HPP */
