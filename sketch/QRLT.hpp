#ifndef SKYLARK_QRLT_HPP
#define SKYLARK_QRLT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Random Laplace Transform
 *
 * Sketch transform into Eucledian space of fuctions in an RKHS
 * implicitly defined by a vector and a semigroup kernel.
 *
 * See:
 *
 * Jiyan Yang, Vikas Sindhwani, Quanfu Fan, Haim Avron, Michael Mahoney
 * Random Laplace Feature Maps for Semigroup Kernels on Histograms
 * CVPR 2014
 *
 * Yang, Sindhawni, Avron and Mahoney
 * Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels
 * ICML 2014
 *
 */
template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename, typename> class KernelDistribution,
           template <typename> class QMCSequenceType>
struct QRLT_t : public QRLT_data_t<KernelDistribution, QMCSequenceType> {
    // To be specilized and derived.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;
    typedef QRLT_data_t<KernelDistribution, QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;

    QRLT_t(int N, int S,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRLT"));
    }

    QRLT_t(const data_type& other_data)
        : data_type(other_data) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRLT"));
    }

    QRLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRLT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRLT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for QRLT"));
    }
};

/**
 * Quasi Random Features for Exponential Semigroup
 */
template< typename InputMatrixType, typename OutputMatrixType,
          template <typename> class QMCSequenceType>
struct ExpSemigroupQRLT_t :
        public ExpSemigroupQRLT_data_t<QMCSequenceType>,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

    // We use composition to defer calls to RLT_t
    typedef QRLT_t<InputMatrixType, OutputMatrixType,
                   internal::levy_distribution_t,
                   QMCSequenceType> transform_t; 

    typedef ExpSemigroupQRLT_data_t<QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;
    typedef typename data_type::params_t params_t;

    ExpSemigroupQRLT_t(int N, int S, double beta,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, beta, sequence, skip, context), _transform(*this) {

    }

    ExpSemigroupQRLT_t(int N, int S, const params_t& params,
        base::context_t& context)
        : data_type(N, S, params, context), _transform(*this) {

    }

    ExpSemigroupQRLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    ExpSemigroupQRLT_t(const ExpSemigroupQRLT_t<OtherInputMatrixType,
        OtherOutputMatrixType, QMCSequenceType>& other)
        : data_type(other), _transform(*this) {

    }

    ExpSemigroupQRLT_t (const data_type& other)
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

/**** Now the implementations */
#include "QRLT_Elemental.hpp"

/**** Now the any,any implementations */
namespace skylark { namespace sketch {

template<template <typename> class QMCSequenceType>
class ExpSemigroupQRLT_t<boost::any, boost::any, QMCSequenceType> :
        public ExpSemigroupQRLT_data_t<QMCSequenceType>,
        virtual public sketch_transform_t<boost::any, boost::any> {

public:

    typedef ExpSemigroupQRLT_data_t<QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;
    typedef typename data_type::params_t params_t;

    ExpSemigroupQRLT_t(int N, int S, double beta,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type(N, S, beta, sequence, skip, context) {

    }

    ExpSemigroupQRLT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context) {

    }


    ExpSemigroupQRLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    ExpSemigroupQRLT_t (const ExpSemigroupQRLT_t<OtherInputMatrixType, 
        OtherOutputMatrixType, QMCSequenceType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    ExpSemigroupQRLT_t (const data_type& other)
        : data_type(other) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply(const boost::any &A, const boost::any &sketch_of_A,
                columnwise_tag dimension) const {

#if     !(defined SKYLARK_NO_ANY)

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::matrix_t, mdtypes::matrix_t,
            ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vc_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vc_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vr_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vr_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_t, ExpSemigroupQRLT_t);

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::matrix_t, mftypes::matrix_t,
            ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::sparse_matrix_t,
            mftypes::matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::root_matrix_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vc_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vc_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vr_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vr_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_t, ExpSemigroupQRLT_t);

#endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for ExpSemigroupQRLT"));

    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const boost::any &A, const boost::any &sketch_of_A,
        rowwise_tag dimension) const {

#if     !(defined SKYLARK_NO_ANY)

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::matrix_t, mdtypes::matrix_t,
            ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vc_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vc_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vr_t,
            mdtypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vr_t,
            mdtypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_t, ExpSemigroupQRLT_t);

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::matrix_t, mftypes::matrix_t,
            ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::sparse_matrix_t,
            mftypes::matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::root_matrix_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vc_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vc_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vr_t,
            mftypes::root_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vr_t,
            mftypes::shared_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_t, ExpSemigroupQRLT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH_QMC(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_t, ExpSemigroupQRLT_t);

#endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for ExpSemigroupQRLT"));
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_QRLT_HPP
