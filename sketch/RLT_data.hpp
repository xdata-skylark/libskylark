#ifndef SKYLARK_RLT_DATA_HPP
#define SKYLARK_RLT_DATA_HPP

#include <vector>

#include "../base/context.hpp"
#include "transform_data.hpp"
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
//FIXME: WHY is this not derived from dense_transform_t??
template <typename ValueType,
          template <typename> class KernelDistribution>
struct RLT_data_t : public transform_data_t {

    typedef ValueType value_type;
    typedef skylark::sketch::dense_transform_data_t<value_type,
                                                    KernelDistribution>
        underlying_data_type;
    typedef transform_data_t base_t;

    /**
     * Regular constructor
     */
    RLT_data_t (int N, int S, skylark::base::context_t& context,
                std::string type)
        : base_t(N, S, context, type), _val_scale(1), _ctx(base_t::build()),
          _underlying_data(base_t::_N, base_t::_S, _ctx),
          _scale(std::sqrt(1.0 / base_t::_S)) {
    }

    RLT_data_t (boost::property_tree::ptree &json)
        : base_t(json), _val_scale(1), _ctx(base_t::build()),
          _underlying_data(base_t::_N, base_t::_S, _ctx),
          _scale(std::sqrt(1.0 / base_t::_S)) {
    }

protected:
    value_type _val_scale; /**< Bandwidth (sigma)  */
    const base::context_t _ctx;
    const underlying_data_type _underlying_data;
    /**< Data of the underlying dense transformation */
    const value_type _scale; /** Scaling for trigonometric factor */
};

/**
 * Random Features for Exponential Semigroup
 */
template<typename ValueType>
struct ExpSemigroupRLT_data_t :
        public RLT_data_t<ValueType, utility::standard_levy_distribution_t> {

    typedef RLT_data_t<ValueType, utility::standard_levy_distribution_t > base_t;

    /**
     * Constructor
     */
    ExpSemigroupRLT_data_t(int N, int S, typename base_t::value_type beta,
        skylark::base::context_t& context)
        : base_t(N, S, context, "ExpSemigroupRLT"), _beta(beta) {

        base_t::_val_scale = beta * beta / 2;
        base_t::build();
    }

    ExpSemigroupRLT_data_t(boost::property_tree::ptree &json)
        : base_t(json),
        _beta(json.get<ValueType>("sketch.beta")) {

        base_t::_val_scale = _beta * _beta / 2;
        base_t::build();
    }

    template <typename ValueT>
    friend boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const ExpSemigroupRLT_data_t<ValueT> &data);

protected:
    const ValueType _beta;
};

template <typename ValueType>
boost::property_tree::ptree& operator<<(
        boost::property_tree::ptree &sk,
        const ExpSemigroupRLT_data_t<ValueType> &data) {

    sk << static_cast<const transform_data_t&>(data);
    sk.put("sketch.beta", data._beta);
    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RLT_DATA_HPP */
