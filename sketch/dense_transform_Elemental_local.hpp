#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_LOCAL_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_LOCAL_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/get_communicator.hpp"

#include "sketch_params.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization local input (sparse of dense), local output.
 * InputType should either be El::Matrix, or base:sparse_matrix_t.
 */
template <typename ValueType,
          template <typename> class InputType,
          typename ValuesAccessor>
struct dense_transform_t <
    InputType<ValueType>,
    El::Matrix<ValueType>,
    ValuesAccessor> :
        public dense_transform_data_t<ValuesAccessor> {

    typedef ValueType value_type;
    typedef InputType<value_type> matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;
    typedef dense_transform_data_t<ValuesAccessor> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, double scale, base::context_t& context)
        : data_type (N, S, scale, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (const dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValuesAccessor>& other)
        : data_type(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const data_type& other_data)
        : data_type(other_data) {}


    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl_local(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                base::elemental_exception()
                    << base::error_msg(e.what()) );
        }
    }

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

    const sketch_transform_data_t* get_data() const { return this; }

private:

    // TODO: Block-by-block mode
    void apply_impl_local (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag tag) const {

        output_matrix_type R;
        data_type::realize_matrix_view(R);

        base::Gemm (El::NORMAL,
                    El::TRANSPOSE,
                    value_type(1),
                    A,
                    R,
                    value_type(0),
                    sketch_of_A);
    }


    // TODO: Block-by-block mode
    void apply_impl_local (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag tag) const {

        output_matrix_type R;
        data_type::realize_matrix_view(R);

        base::Gemm (El::NORMAL,
                    El::NORMAL,
                    value_type(1),
                    R,
                    A,
                    value_type(0),
                    sketch_of_A);
    }
};

template <typename ValueType,
          typename ValuesAccessor>
struct dense_transform_t <
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    ValuesAccessor> :
        public dense_transform_data_t<ValuesAccessor> {

    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> output_matrix_type;
    typedef dense_transform_data_t<ValuesAccessor> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, double scale, base::context_t& context)
        : data_type (N, S, scale, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (const dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValuesAccessor>& other)
        : data_type(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const data_type& other_data)
        : data_type(other_data) {}


    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl_local(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                base::elemental_exception()
                    << base::error_msg(e.what()) );
        }
    }

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

    const sketch_transform_data_t* get_data() const { return this; }

private:

    // TODO: Block-by-block mode
    void apply_impl_local (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag tag) const {

        El::Matrix<ValueType> R;
        data_type::realize_matrix_view(R);

        base::Gemm (El::NORMAL,
                    El::TRANSPOSE,
                    value_type(1),
                    A.LockedMatrix(),
                    R,
                    value_type(0),
                    sketch_of_A.Matrix());
    }


    // TODO: Block-by-block mode
    void apply_impl_local (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag tag) const {

        El::Matrix<ValueType> R;
        data_type::realize_matrix_view(R);

        base::Gemm (El::NORMAL,
                    El::NORMAL,
                    value_type(1),
                    R,
                    A.LockedMatrix(),
                    value_type(0),
                    sketch_of_A.Matrix());
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_LOCAL_HPP
