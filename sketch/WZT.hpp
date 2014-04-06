#ifndef SKYLARK_WZT_HPP
#define SKYLARK_WZT_HPP

#include <boost/random.hpp>
#include "WZT_data.hpp"
#include "hash_transform.hpp"

namespace skylark { namespace sketch {

/**
 * Woodruff-Zhang Transform (data)
 *
 * Woodruff-Zhang Transform is very similar to the Clarkson-Woodruff Transform:
 * it replaces the +1/-1 diagonal with reciprocal exponentia random enteries. 
 * It is sutiable for lp regression with 1 <= p <= 2.
 *
 * Reference:
 * D. Woodruff and Q. Zhang
 * Subspace Embeddings and L_p Regression Using Exponential Random
 * COLT 2013
 *
 * TODO current implementation is only one sketch index, when for 1 <= p <= 2
 *      you want more than one.
 */

#define _SL_HTBASE hash_transform_t< InputMatrixType, OutputMatrixType, \
                                     boost::random::uniform_int_distribution, \
                                     boost::random::exponential_distribution >
template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct WZT_t :
        public WZT_data_t< typename _SL_HTBASE::index_type,
                           typename _SL_HTBASE::value_type > {
public:

    // We use composition to defer calls to hash_transform_t
    typedef _SL_HTBASE transform_t;

    typedef WZT_data_t< typename _SL_HTBASE::index_type,
                        typename _SL_HTBASE::value_type > data_type;

    /**
     * Regular constructor
     */
    WZT_t(int N, int S, double p, context_t& context)
        : data_type(N, S, p, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template< typename OtherInputMatrixType,
              typename OtherOutputMatrixType >
    WZT_t(const WZT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    WZT_t(const data_type& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const typename transform_t::matrix_type& A,
                typename transform_t::output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        _transform.apply(A, sketch_of_A, dimension);
    }

private:
    transform_t _transform;
};

#undef _SL_HTBASE

} } /** namespace skylark::sketch */

#endif // SKYLARK_WZT_HPP
