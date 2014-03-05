#ifndef SKYLARK_JLT_HPP
#define SKYLARK_JLT_HPP

#include "JLT_data.hpp"
#include "dense_transform.hpp"

namespace skylark { namespace sketch {

namespace bstrand = boost::random;

/**
 * Johnson-Lindenstrauss Transform
 *
 * The JLT is simply a dense random matrix with i.i.d normal entries.
 */
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
class JLT_t :
  public JLT_data_t<typename
    dense_transform_t<InputMatrixType, OutputMatrixType,
                      bstrand::normal_distribution >::value_type >,
  virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

public:

    // We use composition to defer calls to dense_transform_t
    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
                               bstrand::normal_distribution > transform_t;

    typedef JLT_data_t<typename transform_t::value_type> base_t;

    /**
     * Regular constructor
     */
    JLT_t(int N, int S, skylark::sketch::context_t& context)
        : base_t(N, S, context), _transform(*this) {

    }

    JLT_t(boost::property_tree::ptree &json,
          skylark::sketch::context_t& context)
        : base_t(json, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    JLT_t (const JLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : base_t(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    JLT_t (const base_t& other)
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

    int get_N() const { return this->N; } /**< Get input dimesion. */
    int get_S() const { return this->S; } /**< Get output dimesion. */

private:
    transform_t _transform;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_JLT_HPP
