#ifndef SKYLARK_QRLT_DATA_HPP
#define SKYLARK_QRLT_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <boost/math/special_functions/erf.hpp>
#include <vector>

#include "../utility/randgen.hpp"
#include "../utility/quasirand.hpp"

namespace skylark { namespace sketch {


/**
 * Random Laplace Transform (data)
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a semigroup kernel.
 *
 * Use quasi-random features.
 *
 * See:
 *
 * Jiyan Yang, Vikas Sindhwani, Quanfu Fan, Haim Avron, Michael Mahoney
 * Random Laplace Feature Maps for Semigroup Kernels on Histograms
 * CVPR 2014
 *
 * Yang, Sindhawni, Avron and Mahoney
 * Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels
 * ICML 2014
 *
 */
template <template <typename, typename> class KernelDistribution,
          template <typename> class QMCSequenceType>
struct QRLT_data_t : public sketch_transform_data_t {

    typedef double value_type;
    typedef quasi_dense_transform_data_t<KernelDistribution, QMCSequenceType>
    underlying_data_type;
    typedef QMCSequenceType<value_type> sequence_type;
    typedef sketch_transform_data_t base_t;

    static size_t qmc_sequence_dim(size_t N)  { return N; }

    QRLT_data_t (int N, int S, double inscale, double outscale,
        const sequence_type& sequence, int skip,  base::context_t& context)
        : base_t(N, S, context, "QRLT"), _inscale(inscale),
          _outscale(outscale), _sequence(sequence), _skip(skip) {

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


    QRLT_data_t (int N, int S, double inscale, double outscale,
        const sequence_type& sequence, int skip,
        const base::context_t& context, std::string type)
        : base_t(N, S, context, type), _inscale(inscale),
          _outscale(outscale), _sequence(sequence), _skip(skip) {

    }

   base::context_t build() {
       base::context_t ctx = base_t::build();
       _underlying_data = boost::shared_ptr<underlying_data_type>(new
           underlying_data_type(base_t::_N, base_t::_S, _inscale, 
               _sequence, _skip, ctx));
       return ctx;
   }

    double _inscale;
    double _outscale; /** Scaling for exponential factor */
    boost::shared_ptr<underlying_data_type> _underlying_data;
    /**< Data of the underlying dense transformation */
    sequence_type _sequence;
    const int _skip;
};

namespace internal {

// There is no boost::math class for levy's distribution.
// However, we need only the quantile function, which 
// we implement in this skeleton.

template <class ValueType = double, 
          class Policy = boost::math::policies::policy<> >
struct levy_distribution_t
{
    typedef ValueType value_type;
    typedef Policy policy_type;

    levy_distribution_t(value_type mean = 0, value_type scale = 1)
        : _mean(mean), _scale(scale) {

    }

    value_type get_mean() const {
        return _mean;
    }

    value_type get_scale() const {
        return _scale;
    }

private:
    value_type _mean;
    value_type _scale;
};

// I have no idea how the compiler/boost finds this function!
// It works on my machine, but I hope it will not break in others...
template <class RealType, class Policy>
inline RealType quantile(const internal::levy_distribution_t<RealType,
    Policy>& dist,
    const RealType& p) {
    RealType v = boost::math::erfc_inv(p, Policy());
    return dist.get_scale() / (2 * v * v) + dist.get_mean();
}

}

/**
 * Quasi Random Features for Exponential Semigroup
 */
template<template <typename> class QMCSequenceType>
struct ExpSemigroupQRLT_data_t :
        public QRLT_data_t<internal::levy_distribution_t,
                           QMCSequenceType> {

    typedef QRLT_data_t<internal::levy_distribution_t,
                        QMCSequenceType> base_t;

    typedef typename base_t::sequence_type sequence_type;

    /// Params structure
    struct params_t : public sketch_params_t {

        params_t(double beta, const sequence_type& sequence, int skip) :
            beta(beta), sequence(sequence), skip(skip) {

        }

        const double beta;
        const sequence_type sequence;
        const int skip;
    };

    ExpSemigroupQRLT_data_t(int N, int S, double beta,
        const sequence_type& sequence, int skip,
        base::context_t& context)
        : base_t(N, S, beta * beta / 2, std::sqrt(1.0 / S),
            sequence, skip, context, "ExpSemigroupQRLT"),
          _beta(beta), _sequence(sequence), _skip(skip) {

        context = base_t::build();
    }

    ExpSemigroupQRLT_data_t(int N, int S, const params_t& params,
        base::context_t& context)
        : base_t(N, S, params.beta * params.beta / 2, std::sqrt(1.0 / S),
            params.sequence, params.skip, context, "ExpSemigroupQRLT"),
          _beta(params.beta), _sequence(params.sequence),
          _skip(params.skip) {

        context = base_t::build();
    }

    ExpSemigroupQRLT_data_t(const boost::property_tree::ptree &pt) :
        base_t(pt.get<int>("N"), pt.get<int>("S"),
            pt.get<double>("beta") * pt.get<double>("beta") / 2,
            std::sqrt(1.0 / pt.get<double>("S")),
            sequence_type(pt.get_child("sequence")), pt.get<int>("skip"),
            base::context_t(pt.get_child("creation_context")), "ExpSemiGroupQRLT"),
        _beta(pt.get<double>("beta")),
        _sequence(pt.get_child("sequence")),  _skip(pt.get<int>("skip")) {

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
        pt.put_child("sequence", _sequence.to_ptree());
        pt.put("beta", _beta);
        pt.put("skip", _skip);
        return pt;
    }

protected:
    ExpSemigroupQRLT_data_t(int N, int S, double beta,
        const sequence_type& sequence, int skip,
        const skylark::base::context_t& context, std::string type)
        : base_t(N, S,  beta * beta / 2, std::sqrt(1.0 / S), context, type), 
          _beta(beta), _sequence(sequence), _skip(skip) {

    }


private:
    const double _beta;
    const sequence_type _sequence;
    const int _skip;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_QRLT_DATA_HPP */
