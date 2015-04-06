#ifndef SKYLARK_QRFT_DATA_HPP
#define SKYLARK_QRFT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

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
 *
 * Yang, Sindhawni, Avron and Mahoney
 * Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels
 * ICML 2014
 */
template <template <typename, typename> class KernelDistribution,
          template <typename> class QMCSequenceType>
struct QRFT_data_t : public sketch_transform_data_t {

    typedef double value_type;
    typedef
    quasi_dense_transform_data_t<KernelDistribution, QMCSequenceType>
    underlying_data_type;
    typedef QMCSequenceType<value_type> sequence_type;
    typedef sketch_transform_data_t base_t;

    static size_t qmc_sequence_dim(size_t N)  { return N+1; }

    QRFT_data_t (int N, int S, double inscale, double outscale,
        const sequence_type& sequence, int skip, base::context_t& context)
        : base_t(N, S, context, "QRFT"), _inscale(inscale),
          _outscale(outscale), _shifts(S), _sequence(sequence), _skip(skip) {

        context = build();
    }

    /**
     *  Serializes a sketch to a string.
     *
     *  @param[out] property_tree describing the sketch.
     */
    virtual boost::property_tree::ptree to_ptree() const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Do not yet support serialization of generic QRFT transform"));

        return boost::property_tree::ptree();
    }

    virtual sketch_transform_t<boost::any, boost::any> *get_transform() {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Trying to create concrete transform of QRFT_data_t"));

        return nullptr;
    }

protected:

    typedef typename underlying_data_type::value_accessor_type accessor_type;

    QRFT_data_t (int N, int S, double inscale, double outscale,
        const sequence_type& sequence, int skip,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type),  _inscale(inscale),
          _outscale(outscale), _shifts(S), _sequence(sequence), _skip(skip)  {

    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        _underlying_data = boost::shared_ptr<underlying_data_type>(new
            underlying_data_type(base_t::_N, base_t::_S, _inscale,
                _sequence, _skip, ctx));

        const double pi = boost::math::constants::pi<double>();
        for(int i = 0; i < base_t::_S; i++)
            _shifts[i] =  2 * pi * _sequence.coordinate(_skip + i, base_t::_N);

        return ctx;
    }

    double _inscale;
    double _outscale; /** Scaling for trigonometric factor */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */
    std::vector<double> _shifts; /** Shifts for scaled trigonometric factor */
    sequence_type _sequence;
    const int _skip;
};

template<template <typename> class QMCSequenceType>
struct GaussianQRFT_data_t :
        public QRFT_data_t<boost::math::normal_distribution,
                           QMCSequenceType> {

    typedef QRFT_data_t<boost::math::normal_distribution,
                        QMCSequenceType> base_t;

    typedef typename base_t::sequence_type sequence_type;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma, const sequence_type& sequence, int skip) :
            sigma(sigma), sequence(sequence), skip(skip) {

        }

        const double sigma;
        const sequence_type sequence;
        const int skip;
    };

    GaussianQRFT_data_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip,
        base::context_t& context)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            sequence, skip, context, "GaussianQRFT"),
          _sigma(sigma), _sequence(sequence), _skip(skip) {

        context = base_t::build();
    }

    GaussianQRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S),
            params.sequence, params.skip, context, "GaussianQRFT"),
          _sigma(params.sigma), _sequence(params.sequence),
          _skip(params.skip) {

        context = base_t::build();
    }

    GaussianQRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("sigma"),
            std::sqrt(2.0 / pt.get<double>("S")),
            sequence_type(pt.get_child("sequence")), pt.get<int>("skip"),
            base::context_t(pt.get_child("creation_context")), "GaussianQRFT"),
        _sigma(pt.get<double>("sigma")),
        _sequence(pt.get_child("sequence")),  _skip(pt.get<int>("skip")) {

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
        pt.put_child("sequence", _sequence.to_ptree());
        pt.put("sigma", _sigma);
        pt.put("skip", _skip);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform();

protected:
    GaussianQRFT_data_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            sequence, skip, context, type),
          _sigma(sigma), _sequence(sequence), _skip(skip) {

    }

private:
    const double _sigma;
    const sequence_type _sequence;
    const int _skip;
};

template<template <typename> class QMCSequenceType>
struct LaplacianQRFT_data_t :
        public QRFT_data_t<boost::math::cauchy_distribution,
                           QMCSequenceType> {

    typedef QRFT_data_t<boost::math::cauchy_distribution,
                        QMCSequenceType> base_t;

    typedef typename base_t::sequence_type sequence_type;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double sigma, const sequence_type& sequence, int skip) :
            sigma(sigma), sequence(sequence), skip(skip) {

        }

        const double sigma;
        const sequence_type sequence;
        const int skip;
    };

    LaplacianQRFT_data_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip,
        base::context_t& context)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            sequence, skip, context, "LaplacianQRFT"),
        _sigma(sigma), _sequence(sequence), _skip(skip) {

        context = base_t::build();
    }

    LaplacianQRFT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, 1.0 / params.sigma, std::sqrt(2.0 / S),
            params.sequence, params.skip, context, "LaplacianQRFT"),
          _sigma(params.sigma), _sequence(params.sequence),
          _skip(params.skip) {

        context = base_t::build();
    }

    LaplacianQRFT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            1.0 / pt.get<double>("sigma"),
            std::sqrt(2.0 / pt.get<double>("S")),
            sequence_type(pt.get_child("sequence")), pt.get<int>("skip"),
            base::context_t(pt.get_child("creation_context")), "LaplacianQRFT"),
        _sigma(pt.get<double>("sigma")),
        _sequence(pt.get_child("sequence")),  _skip(pt.get<int>("skip")) {

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
        pt.put_child("sequence", _sequence.to_ptree());
        pt.put("sigma", _sigma);
        pt.put("skip", _skip);
        return pt;
    }

    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform();

protected:

    LaplacianQRFT_data_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip,
        const base::context_t& context, std::string type)
        : base_t(N, S, 1.0 / sigma, std::sqrt(2.0 / S),
            sequence, skip, context, type),
          _sigma(sigma), _sequence(sequence), _skip(skip) {

    }

private:
    const double _sigma;
    const sequence_type _sequence;
    const int _skip;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_QRFT_DATA_HPP */
