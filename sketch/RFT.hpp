#ifndef SKYLARK_RFT_HPP
#define SKYLARK_RFT_HPP

#include "RFT_data.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Random Fourier Transform
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a shift-invariant kernel.
 *
 * See:
 * Ali Rahimi and Benjamin Recht
 * Random Features for Large-Scale Kernel Machines
 * NIPS 2007.
 */
template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class KernelDistribution>
class RFT_t {
    // To be specilized and derived.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef RFT_data_t<KernelDistribution> data_type;

    RFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for RFT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for RFT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for RFT"));
    }

private:
    RFT_t(int N, int S, double sigma, base::context_t& context);
};

/**
 * Random Features for Gaussian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType = InputMatrixType>
struct GaussianRFT_t :
    public GaussianRFT_data_t,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {


    // We use composition to defer calls to RFT_t
    typedef RFT_t<InputMatrixType, OutputMatrixType,
                  bstrand::normal_distribution > transform_t;

    typedef GaussianRFT_data_t data_type;

    /**
     * Regular constructor
     */
    GaussianRFT_t(int N, int S, double sigma, base::context_t& context)
        : data_type(N, S, sigma, context), _transform(*this) {

    }

    GaussianRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    GaussianRFT_t(
        const GaussianRFT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    GaussianRFT_t (const data_type& other)
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

/**
 * Random Features for Laplacian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType = InputMatrixType>
struct LaplacianRFT_t :
    public LaplacianRFT_data_t,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {


    // We use composition to defer calls to RFT_t
    typedef RFT_t<InputMatrixType, OutputMatrixType,
                  bstrand::cauchy_distribution > transform_t;

    typedef LaplacianRFT_data_t data_type;

    /**
     * Regular constructor
     */
    LaplacianRFT_t(int N, int S, double sigma, base::context_t& context)
        : data_type(N, S, sigma, context), _transform(*this) {

    }

    LaplacianRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    LaplacianRFT_t(
        const LaplacianRFT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    LaplacianRFT_t (const data_type& other)
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

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    transform_t _transform;

};

} } /** namespace skylark::sketch */


#if SKYLARK_HAVE_ELEMENTAL
#include "RFT_Elemental.hpp"
#endif

#endif // SKYLARK_RFT_HPP
