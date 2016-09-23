#ifndef SKYLARK_CT_HPP
#define SKYLARK_CT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Cauchy Transform
 *
 * The CT is simply a dense random matrix with i.i.d Cauchy variables
 */
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct CT_t :
  public CT_data_t,
  virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {


    // We use composition to defer calls to dense_transform_t
    typedef boost::random::cauchy_distribution<double> distribution_type;
    typedef base::random_samples_array_t<distribution_type> accessor_type;
    typedef dense_transform_t<InputMatrixType, OutputMatrixType, accessor_type>
    transform_t;

    typedef CT_data_t data_type;
    typedef data_type::params_t params_t;

    /**
     * Regular constructor
     */
    CT_t(int N, int S, double C, base::context_t& context)
        : data_type(N, S, C, context), _transform(*this) {

    }

    CT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context),
          _transform(*this) {

    }

    CT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

    }


    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    CT_t (const CT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    CT_t (const data_type& other)
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

template<>
class CT_t<boost::any, boost::any> :
  public CT_data_t,
  virtual public sketch_transform_t<boost::any, boost::any > {

public:

    typedef CT_data_t data_type;
    typedef data_type::params_t params_t;

    CT_t(int N, int S, double C, base::context_t& context)
        : data_type(N, S, C, context) {

    }

    CT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context) {

    }


    CT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    CT_t (const CT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    CT_t (const data_type& other)
        : data_type(other) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply(const boost::any &A, const boost::any &sketch_of_A,
                columnwise_tag dimension) const {

#       if !(defined SKYLARK_NO_ANY)

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t,
            CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_t, CT_t);

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t,
            CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_t, CT_t);

#       endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for CT"));
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const boost::any &A, const boost::any &sketch_of_A,
        rowwise_tag dimension) const {

#       if !(defined SKYLARK_NO_ANY)

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t,
            CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_t, CT_t);

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t,
            CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::root_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::shared_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_t, CT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_t, CT_t);

#       endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for CT"));

    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CT_HPP
