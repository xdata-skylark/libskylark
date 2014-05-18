#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_LOCAL_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_LOCAL_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"


namespace skylark { namespace sketch {

/**
 * Specialization local input (sparse of dense), local output.
 * InputType should either be elem::Matrix, or base:spare_matrix_t.
 */
template <typename ValueType,
          template <typename> class InputType,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    InputType<ValueType>,
    elem::Matrix<ValueType>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {

    typedef ValueType value_type;
    typedef InputType<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                  ValueDistribution> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (const dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValueDistribution>& other)
        : data_type(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const dense_transform_data_t<value_type,
                                            ValueDistribution>& other_data)
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

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for local and columnwise.
     */
    void apply_impl_local(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          columnwise_tag) const {

        elem::Matrix<value_type> S(data_type::_S, data_type::_N);
        for(int j = 0; j < data_type::_N; j++) {
            for (int i = 0; i < data_type::_S; i++) {
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.Set(i, j, data_type::scale * sample);
            }
        }

        base::Gemm (elem::NORMAL,
                    elem::NORMAL,
                    1.0,
                    S,
                    A,
                    0.0,
                    sketch_of_A);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for local and rowwise.
     */
    void apply_impl_local(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          rowwise_tag) const {

        elem::Matrix<value_type> S(data_type::_S, data_type::_N);
        for(int j = 0; j < data_type::_N; j++) {
            for (int i = 0; i < data_type::_S; i++) {
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.Set(i, j, data_type::scale * sample);
            }
        }

        base::Gemm (elem::NORMAL,
                    elem::TRANSPOSE,
                    1.0,
                    A,
                    S,
                    0.0,
                    sketch_of_A);
    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_LOCAL_HPP
