#ifndef SKYLARK_RLT_DATA_HPP
#define SKYLARK_RLT_DATA_HPP

#include <vector>

#include "context.hpp"
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
template <typename ValueType,
          template <typename> class KernelDistribution>
struct RLT_data_t {

    typedef ValueType value_type;
    typedef skylark::sketch::dense_transform_data_t<value_type,
                                                    KernelDistribution>
    underlying_data_type;

    /**
     * Regular constructor
     */
    RLT_data_t (int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _val_scale(1), _context(context),
          _underlying_data(N, S, context),
          _scale(std::sqrt(1.0 / S)) {

    }


    const RLT_data_t& get_data() const {
        return static_cast<const RLT_data_t&>(*this);
    }


protected:
    const int _N; /**< Input dimension  */
    const int _S; /**< Output dimension  */
    value_type _val_scale; /**< Bandwidth (sigma)  */
    skylark::sketch::context_t& _context; /**< Context for this sketch */
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
        skylark::sketch::context_t& context)
        : base_t(N, S, context), _beta(beta) {
        base_t::_val_scale = beta * beta / 2;
    }

protected:
    const ValueType _beta;
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_RLT_DATA_HPP */
