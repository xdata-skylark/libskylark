#ifndef SKYLARK_FRFT_HPP
#define SKYLARK_FRFT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

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
           typename OutputMatrixType = InputMatrixType> 
class FastRFT_t :
    public FastGaussianRFT_data_t,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

    // To be specilized and derived. Just some guards here.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;

    typedef FastRFT_data_t data_type;

    FastRFT_t(int N, int S, base::context_t& context) : data_type(N, S, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for FastRFT"));
    }

    FastRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for FastRFT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for FastRFT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for FastRFT"));
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

/**
 * FastRandom Features for Gaussian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType = InputMatrixType >
struct FastGaussianRFT_t :
    public FastGaussianRFT_data_t,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

    // We use composition to defer calls to RFT_t
    typedef FastRFT_t<InputMatrixType, OutputMatrixType > transform_t;

    typedef FastGaussianRFT_data_t data_type;
    typedef data_type::params_t params_t;

    FastGaussianRFT_t(int N, int S, double sigma, base::context_t& context)
        : data_type(N, S, sigma, context), _transform(*this) {

    }

    FastGaussianRFT_t(int N, int S, const params_t& params,
                          base::context_t& context)
        : data_type(N, S, params, context), _transform(*this) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FastGaussianRFT_t(
        const FastGaussianRFT_t<OtherInputMatrixType, OtherOutputMatrixType> & other)
        : data_type(other), _transform(*this) {

    }

    FastGaussianRFT_t (const data_type& other)
        : data_type(other), _transform(*this) {

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

    const sketch_transform_data_t* get_data() const { return this; }

private:
    transform_t _transform;

};


} } /** namespace skylark::sketch */


#if SKYLARK_HAVE_ELEMENTAL
#include "FRFT_Elemental.hpp"
#endif

#endif // SKYLARK_FRFT_HPP
