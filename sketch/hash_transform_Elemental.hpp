#ifndef SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
#define SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"
#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

template <typename ValueType,
          elem::Distribution ColDist,
          typename IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<int,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef IdxDistributionType idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<int,
                                  ValueType,
                                  IdxDistributionType,
                                  ValueDistribution> base_data_t;
    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, skylark::sketch::context_t& context) :
        base_data_t (N, S, context) {}

    /**
     * Copy constructor
     */
    hash_transform_t (hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       idx_distribution_type,
                                       ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        Dimension dimension) {

        switch(ColDist) {
        case elem::VR:
        case elem::VC:
            try {
                apply_impl_vdist (A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    utility::elemental_exception()
                        << utility::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    utility::mpi_exception()
                        << utility::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                utility::unsupported_matrix_distribution() );
        }
  }


private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl_vdist (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) {

        // Create space to hold local part of SA
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
            sketch_of_A.Width(),
            sketch_of_A.LDim());

        //XXX: newly created matrix is not zeroed!
        elem::Zero(SA_part);

        // Construct Pi * A (directly on the fly)
        for (size_t j = 0; j < A.LocalHeight(); j++) {

            size_t col_idx = A.ColShift() + A.ColStride() * j;

            size_t row_idx      = base_data_t::row_idx[col_idx];
            value_type scale_factor = base_data_t::row_value[col_idx];

            for(size_t i = 0; i < A.LocalWidth(); i++) {
                value_type value = scale_factor * A.GetLocal(j, i);
                SA_part.Update(row_idx, A.RowShift() + i, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::reduce (base_data_t::context.comm,
            SA_part.LockedBuffer(),
            SA_part.MemorySize(),
            sketch_of_A.Buffer(),
            std::plus<value_type>(),
            0);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the row-wise direction of sketching.
     */
    void apply_impl_vdist (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag) {

        // Create space to hold local part of SA
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
            sketch_of_A.Width(),
            sketch_of_A.LDim());

        elem::Zero(SA_part);

        // Construct A * Pi (directly on the fly)
        for (size_t j = 0; j < A.LocalHeight(); ++j) {

            size_t row_idx = A.ColShift() + A.ColStride() * j;

            for(size_t i = 0; i < A.LocalWidth(); ++i) {

                size_t col_idx   = A.RowShift() + A.RowStride() * i;
                size_t new_col_idx = base_data_t::row_idx[col_idx];
                value_type value   =
                    base_data_t::row_value[col_idx] * A.GetLocal(j, i);
                SA_part.Update(row_idx, new_col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::reduce (base_data_t::context.comm,
            SA_part.LockedBuffer(),
            SA_part.MemorySize(),
            sketch_of_A.Buffer(),
            std::plus<value_type>(),
            0);
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
