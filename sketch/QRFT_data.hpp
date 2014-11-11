#ifndef SKYLARK_QRFT_DATA_HPP
#define SKYLARK_QRFT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "../utility/randgen.hpp"
#include "../utility/quasirand.hpp"

namespace skylark { namespace sketch {


/**
 * Quasi Random Fourier Transform (data)
 *
 * Sketch transform into Euclidean space of functions in an RKHS
 * implicitly defined by a vector and a shift-invariant kernel.
 *
 * Use quasi-random features.
 *
 * See:
 * Yang, Sindhawni, Avron and Mahoney
 * Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels
 * ICML 2014
 */
template <template <typename, typename> class KernelDistribution>
struct QRFT_data_t : public sketch_transform_data_t {

    typedef quasi_dense_transform_data_t<KernelDistribution>
    underlying_data_type;
    typedef sketch_transform_data_t base_t;

    QRFT_data_t (int N, int S, double inscale, double outscale,
         base::context_t& context)
        : base_t(N, S, context, "QRFT"), _inscale(inscale),
          _outscale(outscale), _shifts(S),
          leap(boost::math::prime(N + 1)) {

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

    typedef typename underlying_data_type::value_accessor_type accessor_type;

    QRFT_data_t (int N, int S, double inscale, double outscale,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type),  _inscale(inscale),
          _outscale(outscale), _shifts(S),
          leap(boost::math::prime(N + 1)) {

    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        _underlying_data = boost::shared_ptr<underlying_data_type>(new
        underlying_data_type(base_t::_N, base_t::_S, _inscale, skip, leap, ctx));

        const double pi = boost::math::constants::pi<double>();
        for(int i = 0; i < base_t::_S; i++)
            _shifts[i] =  2 * pi * utility::Halton(base_t::_N + 1,
                (skip + i) * leap, base_t::_N);

        return ctx;
    }

    double _inscale;
    double _outscale; /** Scaling for trigonometric factor */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */
    std::vector<double> _shifts; /** Shifts for scaled trigonometric factor */
    const int leap;
    const int skip = 1000;
};

struct GaussianQRFT_data_t :
        public QRFT_data_t<boost::math::normal_distribution> {

    typedef QRFT_data_t<boost::math::normal_distribution > base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma) : sigma(sigma) {

        }

        const double sigma;
    };

    GaussianQRFT_data_t(int N, int S, double sigma,
        base::context_t& context)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, "GaussianRFT"), 
          _sigma(sigma) {

        context = base_t::build();
    }

    GaussianQRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S), context, "GaussianRFT"), 
          _sigma(params.sigma) {

        context = base_t::build();
    }

    GaussianQRFT_data_t(const boost::property_tree::ptree &pt) :
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
    GaussianQRFT_data_t(int N, int S, double sigma,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, type),
          _sigma(sigma) {

    }

private:
    const double _sigma;
};

struct LaplacianQRFT_data_t :
    public QRFT_data_t<boost::math::cauchy_distribution> {

    typedef QRFT_data_t<boost::math::cauchy_distribution > base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma) : sigma(sigma) {

        }

        const double sigma;
    };

    LaplacianQRFT_data_t(int N, int S, double sigma,
        base::context_t& context)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, "LaplacianRFT"),
        _sigma(sigma) {

        context = base_t::build();
    }

    LaplacianQRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S), context, "LaplacianRFT"),
        _sigma(params.sigma) {

        context = base_t::build();
    }

    LaplacianQRFT_data_t(const boost::property_tree::ptree &pt) :
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

    LaplacianQRFT_data_t(int N, int S, double sigma,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S), context, type), 
          _sigma(sigma) {

    }

private:
    const double _sigma;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_QRFT_DATA_HPP */
