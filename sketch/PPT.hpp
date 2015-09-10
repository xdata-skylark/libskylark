#ifndef SKYLARK_PPT_HPP
#define SKYLARK_PPT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

namespace skylark { namespace sketch {

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct PPT_t :
        public PPT_data_t,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {
    // To be specilized and derived. Just some guards here.
    typedef InputMatrixType matrix_type;
    typedef OutputMatrixType output_matrix_type;

    typedef PPT_data_t data_type;
    typedef data_type::params_t params_t;

    PPT_t(int N, int S, int q, double c, double gamma,
        base::context_t& context) :
        data_type(N, S, q, c, gamma, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    PPT_t(int N, int S, const params_t& params, base::context_t& context) :
        data_type(N, S, params, context) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    PPT_t(const data_type& other_data)
        : data_type(other_data) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }
    PPT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

} } 

/**** Now the implementations */
#include "PPT_Elemental.hpp"

/**** Now the any,any implementations */
namespace skylark { namespace sketch {

template<>
class PPT_t<boost::any, boost::any> :
  public PPT_data_t,
  virtual public sketch_transform_t<boost::any, boost::any > {

public:

    typedef PPT_data_t data_type;
    typedef data_type::params_t params_t;

    PPT_t(int N, int S, int q, double c, double gamma, base::context_t& context)
        : data_type(N, S, q, c, gamma, context) {

    }

    PPT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type(N, S, params, context) {

    }


    PPT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    PPT_t (const PPT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    PPT_t (const data_type& other)
        : data_type(other) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply(const boost::any &A, const boost::any &sketch_of_A,
                columnwise_tag dimension) const {


#if     !(defined SKYLARK_NO_ANY) || (defined SKYLARK_WITH_GAUSSIAN_RFT_ANY)

#if SKYLARK_HAVE_FFTW

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t,
            PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, PPT_t);

#endif

#if SKYLARK_HAVE_FFTWF

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t,
            PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
            mftypes::root_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, PPT_t);

#endif

#endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));

    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const boost::any &A, const boost::any &sketch_of_A,
        rowwise_tag dimension) const {

#if     !(defined SKYLARK_NO_ANY) || (defined SKYLARK_WITH_GAUSSIAN_RFT_ANY)

#if SKYLARK_HAVE_FFTW

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::matrix_t, mdtypes::matrix_t,
            PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::sparse_matrix_t,
            mdtypes::matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::shared_matrix_t,
            mdtypes::shared_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::root_matrix_t,
            mdtypes::root_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
            mdtypes::dist_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vc_star_t,
            mdtypes::dist_matrix_vc_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_vr_star_t,
            mdtypes::dist_matrix_vr_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vc_t,
            mdtypes::dist_matrix_star_vc_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_star_vr_t,
            mdtypes::dist_matrix_star_vr_t, PPT_t);

#endif

#if SKYLARK_HAVE_FFTWF

        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::matrix_t, mftypes::matrix_t,
            PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::sparse_matrix_t,
            mftypes::matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::shared_matrix_t,
            mftypes::shared_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::root_matrix_t,
            mftypes::root_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
            mftypes::dist_matrix_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vc_star_t,
            mftypes::dist_matrix_vc_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_vr_star_t,
            mftypes::dist_matrix_vr_star_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vc_t,
            mftypes::dist_matrix_star_vc_t, PPT_t);
        SKYLARK_SKETCH_ANY_APPLY_DISPATCH(mftypes::dist_matrix_star_vr_t,
            mftypes::dist_matrix_star_vr_t, PPT_t);

#endif

#endif

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This combination has not yet been implemented for PPT"));
    }

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }
};

} } /** namespace skylark::sketch */


#endif // SKYLARK_PPT_HPP
