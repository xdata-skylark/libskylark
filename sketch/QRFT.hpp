#ifndef SKYLARK_QRFT_HPP
#define SKYLARK_QRFT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Quasi Random Fourier Transform
 *
 * Sketch transform into Euclidean space of functions in an RKHS
 * implicitly defined by a vector and a shift-invariant kernel.
 *
 * Use quasi-random features.
 *
 * See:
 * Yang, Sindhawni, Avron and Mahoney
 * Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels
 * ICML 2014
 */
template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename, typename> class KernelDistribution,
           template <typename> class QMCSequenceType>
struct QRFT_t : public QRFT_data_t<KernelDistribution, QMCSequenceType> {
    // To be specilized and derived.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef QRFT_data_t<KernelDistribution, QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;

    QRFT_t(int N, int S,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, sequence, skip, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRFT"));
    }

    QRFT_t(const data_type& other_data)
        : data_type(other_data) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRFT"));
    }

    QRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRFT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRFT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRFT"));
    }
};

/**
 * Quasi Random Features for Gaussian Kernel
 */
template< typename InputMatrixType, typename OutputMatrixType,
          template <typename> class QMCSequenceType>
struct GaussianQRFT_t :
        public GaussianQRFT_data_t<QMCSequenceType>,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

    // We use composition to defer calls to QRFT_t
    typedef QRFT_t<InputMatrixType, OutputMatrixType,
                   boost::math::normal_distribution,
                   QMCSequenceType > transform_t;

    typedef GaussianQRFT_data_t<QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;
    typedef typename data_type::params_t params_t;

    GaussianQRFT_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, sigma, sequence, skip, context), _transform(*this) {

    }

    GaussianQRFT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context), _transform(*this) {

    }

    GaussianQRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    GaussianQRFT_t(const GaussianQRFT_t<OtherInputMatrixType,
        OtherOutputMatrixType, QMCSequenceType>& other)
        : data_type(other), _transform(*this) {

    }

    GaussianQRFT_t (const data_type& other)
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
 * Qausi Random Features for Laplacian Kernel
 */
template< typename InputMatrixType, typename OutputMatrixType,
          template <typename> class QMCSequenceType>
struct LaplacianQRFT_t :
        public LaplacianQRFT_data_t<QMCSequenceType>,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

    // We use composition to defer calls to QRFT_t
    typedef QRFT_t<InputMatrixType, OutputMatrixType,
                   boost::math::cauchy_distribution,
                   QMCSequenceType> transform_t;

    typedef LaplacianQRFT_data_t<QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;
    typedef typename data_type::params_t params_t;

    LaplacianQRFT_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip,
        base::context_t& context)
        : data_type(N, S, sigma, sequence, skip, context), _transform(*this) {

    }

    LaplacianQRFT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context), _transform(*this) {

    }

    LaplacianQRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    LaplacianQRFT_t(const LaplacianQRFT_t<OtherInputMatrixType,
        OtherOutputMatrixType, QMCSequenceType>& other)
        : data_type(other), _transform(*this) {

    }

    LaplacianQRFT_t (const data_type& other)
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

template<template <typename> class QMCSequenceType>
class GaussianQRFT_t<boost::any, boost::any, QMCSequenceType> :
        public GaussianQRFT_data_t<QMCSequenceType>,
        virtual public sketch_transform_t<boost::any, boost::any> {

public:

    typedef GaussianQRFT_data_t<QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;
    typedef typename data_type::params_t params_t;

    GaussianQRFT_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, sigma, sequence, skip, context) {

    }

    GaussianQRFT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context) {

    }


    GaussianQRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    GaussianQRFT_t (const GaussianQRFT_t<OtherInputMatrixType, 
        OtherOutputMatrixType, QMCSequenceType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    GaussianQRFT_t (const data_type& other)
        : data_type(other) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply(const boost::any &A, const boost::any &sketch_of_A,
                columnwise_tag dimension) const {
        std::cout << "TODO\n";
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const boost::any &A, const boost::any &sketch_of_A,
        rowwise_tag dimension) const {
        std::cout << "TODO\n";
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

template<template <typename> class QMCSequenceType>
class LaplacianQRFT_t<boost::any, boost::any, QMCSequenceType> :
        public LaplacianQRFT_data_t<QMCSequenceType>,
        virtual public sketch_transform_t<boost::any, boost::any> {

public:

    typedef LaplacianQRFT_data_t<QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;
    typedef typename data_type::params_t params_t;

    LaplacianQRFT_t(int N, int S, double sigma,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, sigma, sequence, skip, context) {

    }

    LaplacianQRFT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context) {

    }


    LaplacianQRFT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    LaplacianQRFT_t (const LaplacianQRFT_t<OtherInputMatrixType,
        OtherOutputMatrixType, QMCSequenceType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    LaplacianQRFT_t (const data_type& other)
        : data_type(other) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply(const boost::any &A, const boost::any &sketch_of_A,
                columnwise_tag dimension) const {
        std::cout << "TODO\n";
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const boost::any &A, const boost::any &sketch_of_A,
        rowwise_tag dimension) const {
        std::cout << "TODO\n";
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

} } /** namespace skylark::sketch */


#include "QRFT_Elemental.hpp"

#endif // SKYLARK_QRFT_HPP
