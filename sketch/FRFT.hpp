#ifndef SKYLARK_FRFT_HPP
#define SKYLARK_FRFT_HPP

#include "FRFT_data.hpp"

namespace skylark { namespace sketch {

/**
 * Fast Random Features Transform
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a shift invariant kernel. Fast variant
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
          typename OutputMatrixType = InputMatrixType >
struct FastGaussianRFT_t :
    public FastGaussianRFT_data_t<typename
        FastRFT_t<InputMatrixType, OutputMatrixType >::value_type >,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

    // We use composition to defer calls to RFT_t
    typedef FastRFT_t<InputMatrixType, OutputMatrixType > transform_t;

    typedef FastGaussianRFT_data_t<typename
      FastRFT_t<InputMatrixType, OutputMatrixType >::value_type > base_t;

    /**
     * Regular constructor
     */
    FastGaussianRFT_t(int N, int S, double sigma,
        skylark::base::context_t& context)
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
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const typename transform_t::matrix_type& A,
                typename transform_t::output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        _transform.apply(A, sketch_of_A, dimension);
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const typename transform_t::matrix_type& A,
                typename transform_t::output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        _transform.apply(A, sketch_of_A, dimension);
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

private:
    transform_t _transform;

};


} } /** namespace skylark::sketch */


#if SKYLARK_HAVE_ELEMENTAL
#include "FRFT_Elemental.hpp"
#endif

#endif // SKYLARK_FRFT_HPP
