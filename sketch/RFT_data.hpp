#ifndef SKYLARK_RFT_DATA_HPP
#define SKYLARK_RFT_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "transform_data.hpp"
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
struct RFT_data_t : public transform_data_t {

    typedef ValueType value_type;
    typedef skylark::sketch::dense_transform_data_t<value_type,
                                                    KernelDistribution>
        underlying_data_type;
    typedef transform_data_t base_t;

    RFT_data_t (int N, int S, skylark::base::context_t& context,
                std::string name = "")
        : base_t(N, S, context, name), _val_scale(1),
          _underlying_data(N, S, base_t::_context),
          _scale(std::sqrt(2.0 / S)) {

        _populate();
    }

    RFT_data_t (boost::property_tree::ptree &json,
                skylark::base::context_t& context)
        : base_t(json, context), _val_scale(1),
          _underlying_data(base_t::_N, base_t::_S, base_t::_context),
          _scale(std::sqrt(2.0 / base_t::_S)) {

        _populate();
    }

    template <typename ValueT,
              template <typename> class KernelDist>
    friend boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const RFT_data_t<ValueT, KernelDist> &data);


protected:
    value_type _val_scale; /**< Bandwidth (sigma)  */
    const underlying_data_type _underlying_data;
    /**< Data of the underlying dense transformation */
    const value_type _scale; /** Scaling for trigonometric factor */
    std::vector<value_type> _shifts; /** Shifts for scaled trigonometric factor */

    void _populate() {

        const double pi = boost::math::constants::pi<value_type>();
        boost::random::uniform_real_distribution<value_type>
            distribution(0, 2 * pi);
        _shifts = base_t::_context.generate_random_samples_array(
            base_t::_S, distribution);
    }
};

template <typename ValueType,
          template <typename> class KernelDistribution>
boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const RFT_data_t<ValueType, KernelDistribution> &data) {

    sk << static_cast<const transform_data_t&>(data);
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
        : base_t(N, S, context, "GaussianRFT"), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
    }

    GaussianRFT_data_t(boost::property_tree::ptree &json,
        skylark::base::context_t& context)
        : base_t(json, context), _sigma(json.get<ValueType>("sketch.sigma")) {
        base_t::_val_scale = 1.0 / _sigma;
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const GaussianRFT_data_t<ValueT> &data);

protected:
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
        : base_t(N, S, context, "LaplacianRFT"), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
    }

    LaplacianRFT_data_t(boost::property_tree::ptree &json,
        skylark::base::context_t& context)
        : base_t(json, context), _sigma(json.get<ValueType>("sketch.sigma")) {
        base_t::_val_scale = 1.0 / _sigma;
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk,
            const LaplacianRFT_data_t<ValueT> &data);

protected:
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
