#ifndef SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
#define SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"
#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization local input, local output
 */
template <typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    elem::Matrix<ValueType>,
    elem::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<size_t,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::Matrix<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<size_t,
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
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const base_data_t& other_data) :
        base_data_t(other_data.get_data()) {}

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        Dimension dimension) const {
        try {
            apply_impl(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::elemental_exception()
                    << utility::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        }
    }


private:

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) const {

        elem::Zero(sketch_of_A);

        // Construct Pi * A (directly on the fly)
        for (size_t row_idx = 0; row_idx < A.Height(); row_idx++) {

            size_t new_row_idx      = base_data_t::row_idx[row_idx];
            value_type scale_factor = base_data_t::row_value[row_idx];

            for(size_t col_idx = 0; col_idx < A.Width(); col_idx++) {
                value_type value = scale_factor * A.Get(row_idx, col_idx);
                sketch_of_A.Update(new_row_idx, col_idx, value);
            }
        }
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the row-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag) const {

        elem::Zero(sketch_of_A);

        // Construct Pi * A (directly on the fly)
        for (size_t col_idx = 0; col_idx < A.Width(); col_idx++) {

            size_t new_col_idx      = base_data_t::row_idx[col_idx];
            value_type scale_factor = base_data_t::row_value[col_idx];

            for(size_t row_idx = 0; row_idx < A.Height(); row_idx++) {
                value_type value = scale_factor * A.Get(row_idx, col_idx);
                sketch_of_A.Update(row_idx, new_col_idx, value);
            }
        }
    }
};

/**
 * Specialization distributed input, local output
 */
template <typename ValueType,
          elem::Distribution ColDist,
          elem::Distribution RowDist,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    elem::DistMatrix<ValueType, ColDist, RowDist>,
    elem::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<size_t,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, RowDist> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<size_t,
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
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const base_data_t& other_data) :
        base_data_t(other_data.get_data()) {}

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        Dimension dimension) const {
        try {
            apply_impl(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::elemental_exception()
                    << utility::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        }
    }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) const {

        // TODO this implementation is communication efficient.
        // Sketching a nxd matrix to sxd will communicate O(sdP)
        // doubles, when you can sometime communicate less:
        // For [MC,MR] or [MR,MC] you need O(sd sqrt(P)).
        // For [*, VC/VR] you need only O(sd).

        // Create space to hold local part of SA
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
            sketch_of_A.Width(),
            sketch_of_A.LDim());

        elem::Zero(SA_part);

        // Construct Pi * A (directly on the fly)
        for (size_t j = 0; j < A.LocalHeight(); j++) {

            size_t row_idx = A.ColShift() + A.ColStride() * j;
            size_t new_row_idx      = base_data_t::row_idx[row_idx];
            value_type scale_factor = base_data_t::row_value[row_idx];

            for(size_t i = 0; i < A.LocalWidth(); i++) {
                size_t col_idx = A.RowShift() + A.RowStride() * i;
                value_type value = scale_factor * A.GetLocal(j, i);
                SA_part.Update(new_row_idx, col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::reduce (base_data_t::_context.comm,
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
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag) const {

        // TODO this implementation is communication efficient.
        // Sketching a nxd matrix to sxd will communicate O(sdP)
        // doubles, when you can sometime communicate less:
        // For [MC,MR] or [MR,MC] you need O(sd sqrt(P)).
        // For [VC/VR, *] you need only O(sd).

        // Create space to hold local part of SA
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
            sketch_of_A.Width(),
            sketch_of_A.LDim());

        elem::Zero(SA_part);

        // Construct A * Pi (directly on the fly)
        for (size_t j = 0; j < A.LocalWidth(); ++j) {

            size_t col_idx = A.RowShift() + A.RowStride() * j;
            size_t new_col_idx = base_data_t::row_idx[col_idx];
            value_type scale_factor = base_data_t::row_value[col_idx];

            for(size_t i = 0; i < A.LocalHeight(); ++i) {
                size_t row_idx   = A.ColShift() + A.ColStride() * i;
                value_type value = scale_factor *  A.GetLocal(i, j);
                SA_part.Update(row_idx, new_col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::reduce (base_data_t::_context.comm,
            SA_part.LockedBuffer(),
            SA_part.MemorySize(),
            sketch_of_A.Buffer(),
            std::plus<value_type>(),
            0);
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
