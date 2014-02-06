#ifndef SKYLARK_FRFT_HPP
#define SKYLARK_FRFT_HPP

#include "FRFT_data.hpp"

namespace skylark { namespace sketch {

/**
 * Fast Random Features Transform
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a kernel. Fast variant
 * (also known as Fastfood).
 *
 * See:
 * Q. Le, T. Sarlos, A. Smola
 * Fastfood - Approximating Kernel Expansions in Loglinear Time
 * ICML 2013.
 */
template < typename InputMatrixType,
           typename OutputMatrixType >
class FastRFT_t {
    // To be specilized and derived.

};

/**
 * FastRandom Features for Gaussian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType>
struct FastGaussianRFT_t :
    public FastGaussianRFT_data_t<typename
      FastRFT_t<InputMatrixType, OutputMatrixType >::value_type > {


    // We use composition to defer calls to RFT_t
    typedef FastRFT_t<InputMatrixType, OutputMatrixType > transform_t;

    typedef FastGaussianRFT_data_t<typename
      FastRFT_t<InputMatrixType, OutputMatrixType >::value_type > base_t;

    /**
     * Regular constructor
     */
    FastGaussianRFT_t(int N, int S, double sigma,
        skylark::sketch::context_t& context)
        : base_t(N, S, sigma, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FastGaussianRFT_t(
        const FastGaussianRFT_t<OtherInputMatrixType, OtherOutputMatrixType> & other)
        : base_t(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    FastGaussianRFT_t (const base_t& other)
        : base_t(other), _transform(*this) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const typename transform_t::matrix_type& A,
                typename transform_t::output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        _transform.apply(A, sketch_of_A, dimension);
    }

private:
    transform_t _transform;

};


} } /** namespace skylark::sketch */


#if SKYLARK_HAVE_ELEMENTAL
#include "FRFT_Elemental.hpp"
#endif

#endif // SKYLARK_FRFT_HPP
