#ifndef SKYLARK_MMT_HPP
#define SKYLARK_MMT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <boost/random.hpp>

namespace skylark { namespace sketch {

/**
 * Meng-Mahoney Transform
 *
 * Meng-Mahoney Transform is very similar to the Clarkson-Woodruff Transform:
 * it replaces the +1/-1 diagonal with Cauchy random enteries. Thus, it
 * provides a low-distortion of l1-norm subspace embedding.
 *
 * See Meng and Mahoney's STOC'13 paper.
 */

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct MMT_t :
        public MMT_data_t,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

public:

    // We use composition to defer calls to hash_transform_t
    typedef hash_transform_t< InputMatrixType, OutputMatrixType,
                              boost::random::uniform_int_distribution,
                              boost::random::cauchy_distribution> transform_t;

    typedef MMT_data_t data_type;
    typedef data_type::params_t params_t;

    MMT_t(int N, int S, base::context_t& context)
        : data_type(N, S, context), _transform(*this) {

    }

    MMT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context),
          _transform(*this) {

    }

    MMT_t(const boost::property_tree::ptree& pt)
        : data_type(pt), _transform(*this) {

    }

    template< typename OtherInputMatrixType,
              typename OtherOutputMatrixType >
    MMT_t(const MMT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    MMT_t(const data_type& other)
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
class MMT_t<boost::any, boost::any> :
  public MMT_data_t,
  virtual public sketch_transform_t<boost::any, boost::any > {

public:

    typedef MMT_data_t data_type;
    typedef data_type::params_t params_t;

    MMT_t(int N, int S, base::context_t& context)
        : data_type(N, S, context) {

    }

    MMT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context) {

    }


    MMT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    MMT_t (const MMT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    MMT_t (const data_type& other)
        : data_type(other) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply(const boost::any &A, const boost::any &sketch_of_A,
                columnwise_tag dimension) const {

#if     !(defined SKYLARK_NO_ANY) || (defined SKYLARK_WITH_MMT_ANY)

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t,
            MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::sparse_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, MMT_t);

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t,
            MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::sparse_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, MMT_t);

#endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for MMT"));

    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const boost::any &A, const boost::any &sketch_of_A,
        rowwise_tag dimension) const {

#if     !(defined SKYLARK_NO_ANY) || (defined SKYLARK_WITH_MMT_ANY)

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t,
            MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::sparse_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, MMT_t);

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t,
            MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::sparse_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::root_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::shared_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, MMT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, MMT_t);
#endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for MMT"));
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

#undef _SL_HTBASE

} } /** namespace skylark::sketch */

#endif // SKYLARK_MMT_HPP
