#ifndef SKYLARK_CWT_HPP
#define SKYLARK_CWT_HPP


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
 *c
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

    // We use composition to defer calls to hash_transform_t
    typedef _SL_HTBASE transform_t;

    typedef CWT_data_t<typename _SL_HTBASE::index_type,
                       typename _SL_HTBASE::value_type> data_type;

    /**
     * Regular constructor
     */
    CWT_t(int N, int S, base::context_t& context)
        : data_type(N, S, context), _transform(*this) {

    }

    CWT_t(const boost::property_tree::ptree &json, base::context_t& context)
        : data_type(json, context), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    CWT_t(const CWT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    CWT_t(const data_type& other)
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

#endif // SKYLARK_CWT_HPP
