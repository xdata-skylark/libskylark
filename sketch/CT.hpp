#ifndef SKYLARK_CT_HPP
#define SKYLARK_CT_HPP

#include "CT_data.hpp"
#include "dense_transform.hpp"

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
  public CT_data_t<typename
    dense_transform_t<InputMatrixType, OutputMatrixType,
                      bstrand::cauchy_distribution >::value_type >,
  virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {


    // We use composition to defer calls to dense_transform_t
    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
                               bstrand::cauchy_distribution > transform_t;

    typedef CT_data_t<typename transform_t::value_type> data_type;


    /**
     * Regular constructor
     */
    CT_t(int N, int S, double C, base::context_t* context)
        : data_type(N, S, C, context), _transform(*this) {

    }

    CT_t(boost::property_tree::ptree &json,
          base::context_t& context)
        : data_type(json, context), _transform(*this) {

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


private:
    transform_t _transform;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CT_HPP
