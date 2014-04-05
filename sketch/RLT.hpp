#ifndef SKYLARK_RLT_HPP
#define SKYLARK_RLT_HPP

#include "RLT_data.hpp"

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
 * Random Laplace Feature Maps for Semigroup Kernels on Histograms
 *
 */
template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class KernelDistribution>
class RLT_t {
    // To be specilized and derived.

};

/**
 * Random Features for Exponential Semigroup
 */
template< typename InputMatrixType,
          typename OutputMatrixType>
struct ExpSemigroupRLT_t :
    public ExpSemigroupRLT_data_t<typename
      RLT_t<InputMatrixType, OutputMatrixType,
            utility::standard_levy_distribution_t >::value_type >,
    virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {


    // We use composition to defer calls to RLT_t
    typedef RLT_t<InputMatrixType, OutputMatrixType,
                  utility::standard_levy_distribution_t > transform_t;

    typedef ExpSemigroupRLT_data_t<typename
      RLT_t<InputMatrixType, OutputMatrixType,
            utility::standard_levy_distribution_t >::value_type > data_type;

    /**
     * Regular constructor
     */
    ExpSemigroupRLT_t(int N, int S, double beta,
        skylark::base::context_t& context)
        : data_type (N, S, beta, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    ExpSemigroupRLT_t(
        const ExpSemigroupRLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    ExpSemigroupRLT_t (const data_type& other)
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


#if SKYLARK_HAVE_ELEMENTAL
#include "RLT_Elemental.hpp"
#endif

#endif // SKYLARK_RLT_HPP
