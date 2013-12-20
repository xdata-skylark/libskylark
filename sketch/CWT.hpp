#ifndef SKYLARK_CWT_HPP
#define SKYLARK_CWT_HPP

#include "../utility/distributions.hpp"
#include "CWT_data.hpp"
#include "hash_transform.hpp"

namespace skylark { namespace sketch {

/**
 * Clarkson-Woodruff Transform
 *
 * Clarkson-Woodruff Transform is essentially the CountSketch
 * sketching originally suggested by Charikar et al.
 * Analysis by Clarkson and Woodruff in STOC 2013 shows that
 * this is sketching scheme can be used to build a subspace embedding.
 *
 * CWT was additionally analyzed by Meng and Mahoney (STOC'13) and is equivalent
 * to OSNAP with s=1.
 */

#define _SL_HTBASE hash_transform_t<InputMatrixType, OutputMatrixType, \
                                    boost::random::uniform_int_distribution, \
                                    utility::rademacher_distribution_t>

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
class CWT_t :
        public CWT_data_t<typename _SL_HTBASE::index_type,
                          typename _SL_HTBASE::value_type> {

public:

    // We use composition to defer calls to dense_transform_t
    typedef _SL_HTBASE transform_type;

    typedef CWT_data_t<typename _SL_HTBASE::index_type,
                       typename _SL_HTBASE::value_type> Base;

    /**
     * Regular constructor
     */
    CWT_t(int N, int S, context_t& context)
        : Base(N, S, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    CWT_t(CWT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
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

#undef _SL_HTBASE

} } /** namespace skylark::sketch */

#endif // SKYLARK_CWT_HPP
