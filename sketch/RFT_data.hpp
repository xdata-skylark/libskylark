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
template <typename ValueType,
          template <typename> class KernelDistribution>
struct RFT_data_t : public sketch_transform_data_t {

    typedef ValueType value_type;
    typedef skylark::sketch::dense_transform_data_t<value_type,
                                                    KernelDistribution>
        underlying_data_type;
    typedef sketch_transform_data_t base_t;

    RFT_data_t (int N, int S, skylark::base::context_t& context,
                std::string type = "RFT")
        : base_t(N, S, context, type), _val_scale(1),
          _underlying_data(nullptr),
          _scale(std::sqrt(2.0 / S)) {

        context = build();
    }

    RFT_data_t (boost::property_tree::ptree &json)
        : base_t(json), _val_scale(1),
          _underlying_data(nullptr),
          _scale(std::sqrt(2.0 / base_t::_S)) {

        build();
    }

    virtual ~RFT_data_t() {
        delete _underlying_data;
    }

    template <typename ValueT,
              template <typename> class KernelDist>
    friend boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const RFT_data_t<ValueT, KernelDist> &data);


protected:
    RFT_data_t (int N, int S, skylark::base::context_t& context,
        std::string type = "RFT", bool nobuild = true)
        : base_t(N, S, context, type), _val_scale(1),
          _underlying_data(nullptr),
          _scale(std::sqrt(2.0 / S)) {

    }

    RFT_data_t (boost::property_tree::ptree &json, bool nobuild)
        : base_t(json), _val_scale(1),
          _underlying_data(nullptr),
          _scale(std::sqrt(2.0 / base_t::_S)) {

    }

    base::context_t build() {

        base::context_t ctx = base_t::build();

        _underlying_data = new underlying_data_type(base_t::_N, base_t::_S,
            ctx);

        const double pi = boost::math::constants::pi<value_type>();
        boost::random::uniform_real_distribution<value_type>
            distribution(0, 2 * pi);
        _shifts = ctx.generate_random_samples_array(base_t::_S, distribution);
        return ctx;
    }

    value_type _val_scale; /**< Bandwidth (sigma)  */
    underlying_data_type *_underlying_data;
    /**< Data of the underlying dense transformation */
    const value_type _scale; /** Scaling for trigonometric factor */
    std::vector<value_type> _shifts; /** Shifts for scaled trigonometric factor */


};

template <typename ValueType,
          template <typename> class KernelDistribution>
boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const RFT_data_t<ValueType, KernelDistribution> &data) {

    sk << static_cast<const sketch_transform_data_t&>(data);
    sk.put("sketch.val_scale", data._val_scale);
    return sk;
}

template<typename ValueType>
struct GaussianRFT_data_t :
        public RFT_data_t<ValueType, bstrand::normal_distribution> {

    typedef RFT_data_t<ValueType, bstrand::normal_distribution > base_t;

    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    GaussianRFT_data_t(int N, int S, typename base_t::value_type sigma,
        skylark::base::context_t& context)
        : base_t(N, S, context, "GaussianRFT", true), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
        context = base_t::build();
    }

    GaussianRFT_data_t(boost::property_tree::ptree &json)
        : base_t(json), _sigma(json.get<ValueType>("sketch.sigma")) {
        base_t::_val_scale = 1.0 / _sigma;
        base_t::build();
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const GaussianRFT_data_t<ValueT> &data);

protected:
    GaussianRFT_data_t(int N, int S, typename base_t::value_type sigma,
        skylark::base::context_t& context, bool nobuild)
        : base_t(N, S, context, "GaussianRFT"), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
    }

    GaussianRFT_data_t(boost::property_tree::ptree &json, bool nobuild)
        : base_t(json), _sigma(json.get<ValueType>("sketch.sigma")) {
        base_t::_val_scale = 1.0 / _sigma;
    }

    const ValueType _sigma;
};

template <typename ValueType>
boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const GaussianRFT_data_t<ValueType> &data) {

    sk << static_cast<const typename GaussianRFT_data_t<ValueType>::base_t&>(data);
    sk.put("sketch.sigma", data._sigma);
    return sk;
}


template<typename ValueType>
struct LaplacianRFT_data_t :
        public RFT_data_t<ValueType, bstrand::cauchy_distribution> {

    typedef RFT_data_t<ValueType, bstrand::cauchy_distribution > base_t;

    LaplacianRFT_data_t(int N, int S, typename base_t::value_type sigma,
        skylark::base::context_t& context)
        : base_t(N, S, context, "LaplacianRFT", false), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
        context = base_t::build();
    }

    LaplacianRFT_data_t(boost::property_tree::ptree &json)
        : base_t(json), _sigma(json.get<ValueType>("sketch.sigma")) {
        base_t::_val_scale = 1.0 / _sigma;
        base_t::build();
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const LaplacianRFT_data_t<ValueT> &data);

protected:

    LaplacianRFT_data_t(int N, int S, typename base_t::value_type sigma,
        skylark::base::context_t& context, bool noread)
        : base_t(N, S, context, "LaplacianRFT", false), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
    }

    LaplacianRFT_data_t(boost::property_tree::ptree &json, bool noread)
        : base_t(json), _sigma(json.get<ValueType>("sketch.sigma")) {
        base_t::_val_scale = 1.0 / _sigma;
    }
    const ValueType _sigma;
};

template <typename ValueType>
boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const LaplacianRFT_data_t<ValueType> &data) {

    sk << static_cast<const typename LaplacianRFT_data_t<ValueType>::base_t&>(data);
    sk.put("sketch.sigma", data._sigma);
    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFT_DATA_HPP */
