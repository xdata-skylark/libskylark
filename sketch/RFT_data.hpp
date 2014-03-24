#ifndef SKYLARK_RFT_DATA_HPP
#define SKYLARK_RFT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "dense_transform_data.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {


/**
 * Random Fourier Transform (data)
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a shift-invariant kernel.
 *
 * See:
 * Ali Rahimi and Benjamin Recht
 * Random Features for Large-Scale Kernel Machines
 * NIPS 2007.
 */
template <typename ValueType,
          template <typename> class KernelDistribution>
struct RFT_data_t {

    typedef ValueType value_type;
    typedef skylark::sketch::dense_transform_data_t<value_type,
                                                    KernelDistribution>
    underlying_data_type;

    /**
     * Regular constructor
     */
    RFT_data_t (int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _val_scale(1), _context(context),
          _underlying_data(N, S, context),
          _scale(std::sqrt(2.0 / S)) {
        const double pi = boost::math::constants::pi<value_type>();
        boost::random::uniform_real_distribution<value_type>
            distribution(0, 2 * pi);
        _shifts = context.generate_random_samples_array(S, distribution);
    }

protected:
    const int _N; /**< Input dimension  */
    const int _S; /**< Output dimension  */
    value_type _val_scale; /**< Bandwidth (sigma)  */
    skylark::sketch::context_t& _context; /**< Context for this sketch */
    const underlying_data_type _underlying_data;
    /**< Data of the underlying dense transformation */
    const value_type _scale; /** Scaling for trigonometric factor */
    std::vector<value_type> _shifts; /** Shifts for scaled trigonometric factor */
};

template<typename ValueType>
struct GaussianRFT_data_t :
        public RFT_data_t<ValueType, bstrand::normal_distribution> {

    typedef RFT_data_t<ValueType, bstrand::normal_distribution > base_t;

    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    GaussianRFT_data_t(int N, int S, typename base_t::value_type sigma,
        skylark::sketch::context_t& context)
        : base_t(N, S, context), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
    }

protected:
    const ValueType _sigma;
};

template<typename ValueType>
struct LaplacianRFT_data_t :
        public RFT_data_t<ValueType, bstrand::cauchy_distribution> {

    typedef RFT_data_t<ValueType, bstrand::cauchy_distribution > base_t;

    /**
     * Constructor
     */
    LaplacianRFT_data_t(int N, int S, typename base_t::value_type sigma,
        skylark::sketch::context_t& context)
        : base_t(N, S, context), _sigma(sigma) {
        base_t::_val_scale = 1.0 / sigma;
    }

protected:
    const ValueType _sigma;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFT_DATA_HPP */
