#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST_STAR_STAR_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST_STAR_STAR_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL
#include "sketch_params.hpp"
#include "dense_transform_Elemental_star_rowdist.hpp"
#endif

namespace skylark { namespace sketch {
/**
 * Specialization: [*, VC/VR] -> [STAR, STAR]
 */
template <typename ValueType,
          elem::Distribution RowDist,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    ValueDistribution > :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> matrix_type;
    typedef elem::DistMatrix<value_type, elem::STAR, elem::STAR>
     output_matrix_type;
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
    dense_transform_t (dense_transform_t<matrix_type,
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

        switch(RowDist) {
        case elem::VR:
        case elem::VC:
            try {
                apply_impl_vdist (A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    base::elemental_exception()
                        << base::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    base::mpi_exception()
                        << base::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                base::unsupported_matrix_distribution() );
        }
    }

private:

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST_STAR_STAR

    /**
     * High-performance implementations
     */

    void apply_impl_vdist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::rowwise_tag tag) const {


        matrix_type sketch_of_A_STAR_RD(A.Height(),
                             data_type::_S);

        dense_transform_t<matrix_type, matrix_type, ValueDistribution>
            transform(*this);

        transform.apply(A, sketch_of_A_STAR_RD, tag);

        sketch_of_A = sketch_of_A_STAR_RD;
    }


    void apply_impl_vdist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::columnwise_tag tag) const {

        matrix_type sketch_of_A_STAR_RD(data_type::_S,
                                        A.Width());

        dense_transform_t<matrix_type, matrix_type, ValueDistribution>
            transform(*this);

        transform.apply(A, sketch_of_A_STAR_RD, tag);

        sketch_of_A = sketch_of_A_STAR_RD;
    }

////////////////////////////////////////////////////////////////////////////////

#else // HP_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST_STAR_STAR

////////////////////////////////////////////////////////////////////////////////

    /**
     * BASE implementations
     */

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [*, VR/VC] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           columnwise_tag) const {

        elem::DistMatrix<value_type,
                         elem::STAR, RowDist>
            sketch_of_A_STAR_RowDist(sketch_of_A.Height(), sketch_of_A.Width());
        elem::Zero(sketch_of_A_STAR_RowDist);


        // Matrix S carries the random samples in the sketching operation S*A.
        // We realize S in parts and compute in a number of local rounds.
        // This ensures handling of cases with a huge S.

        // Max memory assigned to S_part at each round (100 MB by default)
        // TODO: Can we optimize this const for the GEMM that follows?
        const int S_PART_MAX_MEMORY = 100000000;

        int S_height = data_type::_S;
        int S_width = data_type::_N;
        int S_row_num_bytes = S_width * sizeof(value_type);

        // TODO: Guard against the case of S_PART_MAX_MEMORY  < S_row_num_bytes
        int S_part_num_rows = S_PART_MAX_MEMORY / S_row_num_bytes;
        int S_num_rows_consumed = 0;

        while (S_num_rows_consumed < S_height) {
            // Setup S_part S which consists of successive rows in S
            int S_part_height = std::min(S_part_num_rows,
                S_height - S_num_rows_consumed);
            elem::Matrix<value_type> S_part(S_part_height,
                S_width);
            elem::Zero(S_part);
            // Fill S_part with appropriate random samples
            for (int i_loc = 0; i_loc < S_part_height; ++i_loc) {
                int i = S_num_rows_consumed + i_loc;
                for(int j = 0; j < S_width; ++j) {
                    value_type sample =
                        data_type::random_samples[j * data_type::_S + i];
                    S_part.Set(i_loc, j, data_type::scale * sample);
                }
            }
            // Setup a view in sketch_of_A to land the result of S_part*A
            elem::Matrix<value_type> sketch_slice;
            elem::View(sketch_slice, sketch_of_A_STAR_RowDist.Matrix(),
                S_num_rows_consumed, 0,
                S_part_height, sketch_of_A_STAR_RowDist.LocalWidth());
            // Do the multiplication: S_part*A
            base::Gemm (elem::NORMAL,
                elem::NORMAL,
                1.0,
                S_part,
                A.LockedMatrix(),
                0.0,
                sketch_slice);
            S_num_rows_consumed += S_part_height;
        }
        sketch_of_A = sketch_of_A_STAR_RowDist;
    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [*, VR/VC] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          rowwise_tag) const {

        // Redistribute matrix A: [STAR, VC/VR] -> [VC/VR, STAR]
        elem::DistMatrix<value_type, RowDist, elem::STAR> A_RowDist_STAR(A);

        elem::DistMatrix<value_type,
                         RowDist,
                         elem::STAR>
            sketch_of_A_RowDist_STAR(sketch_of_A.Height(), sketch_of_A.Width());
        elem::Zero(sketch_of_A_RowDist_STAR);

        elem::Matrix<value_type> S_local(data_type::_S, data_type::_N);
        for (int j = 0; j < data_type::_N; j++) {
            for (int i = 0; i < data_type::_S; i++) {
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S_local.Set(i, j, data_type::scale * sample);
            }
        }

        // Apply S to the local part of A to get the local part of sketch_of_A.
        base::Gemm(elem::NORMAL,
            elem::TRANSPOSE,
            1.0,
            A_RowDist_STAR.LockedMatrix(),
            S_local,
            sketch_of_A_RowDist_STAR.Matrix());

        sketch_of_A = sketch_of_A_RowDist_STAR;

    }

#endif

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST_STAR_STAR_HPP
