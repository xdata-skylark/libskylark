#ifndef SKYLARK_MMT_HPP
#define SKYLARK_MMT_HPP

#include <boost/random.hpp>
#include "MMT_data.hpp"
#include "hash_transform.hpp"

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

#define _SL_HTBASE hash_transform_t< InputMatrixType, OutputMatrixType, \
                                     boost::random::uniform_int_distribution, \
                                     boost::random::cauchy_distribution >

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct MMT_t :
        public MMT_data_t< typename _SL_HTBASE::index_type,
                           typename _SL_HTBASE::value_type > {
public:

    // We use composition to defer calls to hash_transform_t
    typedef _SL_HTBASE transform_t;

    typedef MMT_data_t< typename _SL_HTBASE::index_type,
                        typename _SL_HTBASE::value_type > base_t;

    /**
     * Regular constructor
     */
    MMT_t(int N, int S, context_t& context)
        : base_t(N, S, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template< typename OtherInputMatrixType,
              typename OtherOutputMatrixType >
    MMT_t(const MMT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
        : base_t(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    MMT_t(const base_t& other)
        : base_t(other), _transform(*this) {

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

#endif // SKYLARK_MMT_HPP
