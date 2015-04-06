#ifndef SKYLARK_RFT_DATA_HPP
#define SKYLARK_RFT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

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

    typedef double value_type;
    typedef random_dense_transform_data_t<KernelDistribution>
    underlying_data_type;
    typedef typename underlying_data_type::distribution_type distribution_type;
    typedef sketch_transform_data_t base_t;

    RFT_data_t (int N, int S, double inscale, double outscale,
        const distribution_type& distribution, base::context_t& context)
        : base_t(N, S, context, "RFT"), _inscale(inscale),
          _outscale(outscale), _scales(S), _distribution(distribution) {

        context = build();
    }

    virtual boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic RFT transform"));

        return boost::property_tree::ptree();
    }

    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Trying to create concrete transform of RFT_data_t"));

        return nullptr;
    }

protected:

    typedef typename underlying_data_type::accessor_type accessor_type;

    RFT_data_t (int N, int S, double inscale, double outscale,
        const distribution_type& distribution,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type),  _inscale(inscale),
          _outscale(outscale), _scales(S), _distribution(distribution) {

    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        _underlying_data = boost::shared_ptr<underlying_data_type>(new
            underlying_data_type(base_t::_N, base_t::_S, _inscale,
                _distribution, ctx));

        const double pi = boost::math::constants::pi<double>();
        boost::random::uniform_real_distribution<double>
            distribution(0, 2 * pi);
        _shifts = ctx.generate_random_samples_array(base_t::_S, distribution);

        // Fill scaling matrix with 1. Subclasses (which are adapted to concrete
        // kernels) could modify this.
        std::fill(_scales.begin(), _scales.end(), 1.0);

        return ctx;
    }

    double _inscale;
    double _outscale; /** Scaling for trigonometric factor */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */
    std::vector<double> _shifts; /** Shifts for scaled trigonometric factor */
    std::vector<double> _scales; /** Scaling based on kernel (filled by subclass) */
    distribution_type _distribution;
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
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            bstrand::normal_distribution<double>(),
            context, "GaussianRFT"),
          _sigma(sigma) {

        context = base_t::build();
    }

    GaussianRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S),
             bstrand::normal_distribution<double>(),
            context, "GaussianRFT"),
          _sigma(params.sigma) {

        context = base_t::build();
    }

    GaussianRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("sigma"),
            std::sqrt(2.0 / pt.get<double>("S")),
             bstrand::normal_distribution<double>(),
            base::context_t(pt.get_child("creation_context")), "GaussianRFT"),
        _sigma(pt.get<double>("sigma")) {

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
        pt.put("sigma", _sigma);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const;

protected:
    GaussianRFT_data_t(int N, int S, double sigma,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
             bstrand::normal_distribution<double>(),
            context, type),
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
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            bstrand::cauchy_distribution<double>(),
            context, "LaplacianRFT"),
        _sigma(sigma) {

        context = base_t::build();
    }

    LaplacianRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S),
            bstrand::cauchy_distribution<double>(),
            context, "LaplacianRFT"),
        _sigma(params.sigma) {

        context = base_t::build();
    }

    LaplacianRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("sigma"),
            std::sqrt(2.0 / pt.get<double>("S")),
            bstrand::cauchy_distribution<double>(),
            base::context_t(pt.get_child("creation_context")), "LaplacianRFT"),
        _sigma(pt.get<double>("sigma")) {

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
        pt.put("sigma", _sigma);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const;

protected:

    LaplacianRFT_data_t(int N, int S, double sigma,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            bstrand::cauchy_distribution<double>(),
            context, type),
          _sigma(sigma) {

    }

private:
    const double _sigma;
};

/**
 * Matern kernel - the sampling probabilities is multivariate-t distribution.
 * See "A Short Review of Multivariate-t Distribution"
 * (Kibria and Joarder, 2006)
 */
struct MaternRFT_data_t :
        public RFT_data_t<bstrand::normal_distribution> {

    typedef RFT_data_t<bstrand::normal_distribution> base_t;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double nu, double l) : nu(nu), l(l) {

        }

        const double nu;
        const double l;
    };

    MaternRFT_data_t(int N, int S, double nu, double l,
        base::context_t& context)
        : base_t(N, S, 1.0 / l, std::sqrt(2.0 / S),
            bstrand::normal_distribution<double>(),
            context, "MaternRFT"),
          _nu(nu), _l(l) {

        context = build();
    }

    MaternRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.l, std::sqrt(2.0 / S),
            bstrand::normal_distribution<double>(),
            context, "MaternRFT"),
          _nu(params.nu), _l(params.l) {

        context = build();
    }

    MaternRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("l"),
            std::sqrt(2.0 / pt.get<double>("S")),
            bstrand::normal_distribution<double>(),
            base::context_t(pt.get_child("creation_context")), "MaternRFT"),
        _nu(pt.get<double>("nu")), _l(pt.get<double>("l")) {

        build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        sketch_transform_data_t::add_common(pt);
        pt.put("nu", _nu);
        pt.put("l", _l);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() const;

protected:
    MaternRFT_data_t(int N, int S, double nu, double l,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / l, std::sqrt(2.0 / S),
            bstrand::normal_distribution<double>(),
            context, type),
          _nu(nu), _l(l) {

    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        boost::random::chi_squared_distribution<double> distribution(2 * _nu);
        _scales = ctx.generate_random_samples_array(base_t::_S, distribution);
        for(auto it = _scales.begin(); it != _scales.end(); it++)
            *it = std::sqrt(2.0 * _nu / *it);

        return ctx;
    }

private:
    const double _nu;
    const double _l;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFT_DATA_HPP */
