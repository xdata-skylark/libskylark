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
                      bstrand::cauchy_distribution >::value_type > {


    // We use composition to defer calls to dense_transform_t
    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
                               bstrand::cauchy_distribution > transform_type;

    typedef CT_data_t<typename transform_type::value_type> Base;


    /**
     * Regular constructor
     */
    CT_t(int N, int S, skylark::sketch::context_t& context)
        : Base(N, S, context), _transform(*this) {
    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    CT_t (const CT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : Base(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    CT_t (const Base& other)
        : Base(other), _transform(*this) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const typename transform_type::matrix_type& A,
                typename transform_type::output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        _transform.apply(A, sketch_of_A, dimension);
    }

private:
    transform_type _transform;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CT_HPP
