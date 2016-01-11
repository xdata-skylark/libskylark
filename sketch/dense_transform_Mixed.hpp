#ifndef SKYLARK_DENSE_TRANSFORM_MIXED_HPP
#define SKYLARK_DENSE_TRANSFORM_MIXED_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/get_communicator.hpp"
#include "../base/sparse_vc_star_matrix.hpp"

#include "sketch_params.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization distributed input sparse_vc_star_matrix_t and output in [VC, *]
 */
template <typename ValueType, typename ValuesAccessor>
struct dense_transform_t <
    base::sparse_vc_star_matrix_t<ValueType>,
    El::DistMatrix<ValueType, El::VC, El::STAR>,
    ValuesAccessor> :
        public dense_transform_data_t<ValuesAccessor> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef base::sparse_vc_star_matrix_t<value_type> matrix_type;
    typedef El::DistMatrix<value_type, El::VC, El::STAR>
    output_matrix_type;
    typedef dense_transform_data_t<ValuesAccessor> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, double scale, base::context_t& context)
        : data_type (N, S, scale, context), _local(*this) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValuesAccessor>& other)
        : data_type(other), _local(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const data_type& other_data)
        : data_type(other_data), _local(other_data) {}

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {

        try {
            apply_impl(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                base::elemental_exception()
                    << base::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
        }
    }

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

    const sketch_transform_data_t* get_data() const { return this; }

private:

    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        _local.apply(A.locked_matrix(), sketch_of_A.Matrix(), tag);

    }

    void apply_impl(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        SKYLARK_THROW_EXCEPTION(base::unsupported_base_operation());

    }

    const dense_transform_t<base::sparse_matrix_t<ValueType>,
                            El::Matrix<ValueType>, ValuesAccessor> _local;

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_MIXED_HPP
