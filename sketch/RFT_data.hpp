#ifndef SKYLARK_RFT_DATA_HPP
#define SKYLARK_RFT_DATA_HPP

#include <vector>

#include "context.hpp"
#include "dense_transform_data.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {


/**
 * Random Features Transform (data)
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a kernel.
 * See:
 * Ali Rahimi and Benjamin Recht
 * Random Features for Large-Scale Kernel Machines
 * NIPS 2007.
 *
 * FIXME control of kernel parameter is not done correctly
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
    RFT_data_t (int N, int S, value_type sigma,
                skylark::sketch::context_t& context)
        : N(N), S(S), sigma(sigma), context(context),
          underlying_data(N, S, context),
          scale(std::sqrt(2.0 / S)) {
        const double pi = boost::math::constants::pi<value_type>();
        boost::random::uniform_real_distribution<value_type>
            distribution(0, 2 * pi);
        shifts = context.generate_random_samples_array(S, distribution);
    }


    const RFT_data_t& get_data() const {
        return static_cast<const RFT_data_t&>(*this);
    }


protected:
    const int N; /**< Input dimension  */
    const int S; /**< Output dimension  */
    const value_type sigma; /**< Bandwidth (sigma)  */
    skylark::sketch::context_t& context; /**< Context for this sketch */
    const underlying_data_type underlying_data;
    /**< Data of the underlying dense transformation */
    const value_type scale; /** Scaling for trigonometric factor */
    std::vector<value_type> shifts; /** Shifts for scaled trigonometric factor */
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
        : base_t(N, S, sigma, context) {

    }

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
        : base_t(N, S, sigma, context) {

    }

};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RFT_DATA_HPP */
