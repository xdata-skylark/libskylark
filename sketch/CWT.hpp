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
        public CWT_data_t,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

public:

    // We use composition to defer calls to hash_transform_t
    typedef hash_transform_t<InputMatrixType, OutputMatrixType,
                             boost::random::uniform_int_distribution,
                             utility::rademacher_distribution_t> transform_t;

    typedef CWT_data_t data_type;

    /**
     * Regular constructor
     */
    CWT_t(int N, int S, base::context_t& context)
        : data_type(N, S, context), _transform(*this) {

    }

    CWT_t(const boost::property_tree::ptree &pt)
        : data_type(pt), _transform(*this) {

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

    const sketch_transform_data_t* get_data() const { return this; }

private:
    transform_t _transform;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_CWT_HPP
