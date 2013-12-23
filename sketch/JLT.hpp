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
                      bstrand::normal_distribution >::value_type > {

public:

    // We use composition to defer calls to dense_transform_t
    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
                               bstrand::normal_distribution > transform_type;

    typedef JLT_data_t<typename transform_type::value_type> Base;

    /**
     * Regular constructor
     */
    JLT_t(int N, int S, skylark::sketch::context_t& context)
        : Base(N, S, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    JLT_t (const JLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : Base(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    JLT_t (const Base& other)
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

#endif // SKYLARK_JLT_HPP
