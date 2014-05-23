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

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct MMT_t :
        public MMT_data_t,
        virtual public sketch_transform_t<InputMatrixType, OutputMatrixType > {

public:

    // We use composition to defer calls to hash_transform_t
    typedef hash_transform_t< InputMatrixType, OutputMatrixType,
                              boost::random::uniform_int_distribution,
                              boost::random::cauchy_distribution> transform_t;

    typedef MMT_data_t data_type;

    /**
     * Regular constructor
     */
    MMT_t(int N, int S, base::context_t& context)
        : data_type(N, S, context), _transform(*this) {

    }

    MMT_t(const boost::property_tree::ptree& pt)
        : data_type(pt), _transform(*this) {

    }

    /**
     * Copy constructor
     */
    template< typename OtherInputMatrixType,
              typename OtherOutputMatrixType >
    MMT_t(const MMT_t<OtherInputMatrixType,OtherOutputMatrixType>& other)
        : data_type(other), _transform(*this) {

    }

    /**
     * Constructor from data
     */
    MMT_t(const data_type& other)
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

#undef _SL_HTBASE

} } /** namespace skylark::sketch */

#endif // SKYLARK_MMT_HPP
