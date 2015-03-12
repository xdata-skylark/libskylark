#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_STAR_STAR_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_STAR_STAR_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/get_communicator.hpp"

#include "sketch_params.hpp"
#include "dense_transform_Elemental_mc_mr.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization distributed input [MC, MR], distributed [STAR, STAR] output
 */
template <typename ValueType,typename ValuesAccessor>
struct dense_transform_t <
    El::DistMatrix<ValueType>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    ValuesAccessor> :
        public dense_transform_data_t<ValuesAccessor> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR>
     output_matrix_type;
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
    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValuesAccessor>& other)
        : data_type(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const dense_transform_data_t<ValuesAccessor>& other_data)
        : data_type(other_data) {}

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl_dist(A, sketch_of_A, dimension);
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


    /**
     * High-performance implementations
     */

    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::rowwise_tag tag) const {


        matrix_type sketch_of_A_MC_MR(A.Height(),
                                  data_type::_S);

        dense_transform_t<matrix_type, matrix_type, ValuesAccessor>
            transform(*this);

        transform.apply(A, sketch_of_A_MC_MR, tag);

        sketch_of_A = sketch_of_A_MC_MR;
    }


    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::columnwise_tag tag) const {


        matrix_type sketch_of_A_MC_MR(data_type::_S,
                                      A.Width());

        dense_transform_t<matrix_type, matrix_type, ValuesAccessor>
            transform(*this);

        transform.apply(A, sketch_of_A_MC_MR, tag);

        sketch_of_A = sketch_of_A_MC_MR;
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_STAR_STAR_HPP
