#ifndef SKYLARK_RFT_HPP
#define SKYLARK_RFT_HPP

#include "RFT_data.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Random Features Transform
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

};

/**
 * Random Features for Gaussian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType>
struct GaussianRFT_t :
    public GaussianRFT_data_t<typename
      RFT_t<InputMatrixType, OutputMatrixType,
            bstrand::normal_distribution >::value_type >,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {


    // We use composition to defer calls to RFT_t
    typedef RFT_t<InputMatrixType, OutputMatrixType,
                  bstrand::normal_distribution > transform_t;

    typedef GaussianRFT_data_t<typename
      RFT_t<InputMatrixType, OutputMatrixType,
            bstrand::normal_distribution >::value_type > base_t;

    /**
     * Regular constructor
     */
    GaussianRFT_t(int N, int S, double sigma,
        skylark::sketch::context_t& context)
        : base_t(N, S, sigma, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    GaussianRFT_t(
        const GaussianRFT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : base_t(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    GaussianRFT_t (const base_t& other)
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

/**
 * Random Features for Laplacian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType>
struct LaplacianRFT_t :
    public LaplacianRFT_data_t<typename
      RFT_t<InputMatrixType, OutputMatrixType,
            bstrand::cauchy_distribution >::value_type > {


    // We use composition to defer calls to RFT_t
    typedef RFT_t<InputMatrixType, OutputMatrixType,
                  bstrand::cauchy_distribution > transform_t;

    typedef LaplacianRFT_data_t<typename
      RFT_t<InputMatrixType, OutputMatrixType,
            bstrand::cauchy_distribution >::value_type > base_t;

    /**
     * Regular constructor
     */
    LaplacianRFT_t(int N, int S, double sigma,
        skylark::sketch::context_t& context)
        : base_t(N, S, sigma, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    LaplacianRFT_t(
        const LaplacianRFT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : base_t(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    LaplacianRFT_t (const base_t& other)
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
#include "RFT_Elemental.hpp"
#endif

#endif // SKYLARK_RFT_HPP
