#ifndef SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
#define SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP

#include <elemental.hpp>
#include "../base/sparse_matrix.hpp"

#include "../base/context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"
#include "../utility/exception.hpp"
#include "../utility/get_communicator.hpp"

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
                                     ValueDistribution>,
        virtual public sketch_transform_t<elem::Matrix<ValueType>,
                                          elem::Matrix<ValueType> > {

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
    hash_transform_t (int N, int S, skylark::base::context_t& context) :
        base_data_t (N, S, context) {}

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const base_data_t& other_data) :
        base_data_t(other_data) {}

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        columnwise_tag dimension) const {
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

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

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
 * Specialization sparse local input, local output
 */
template <typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    base::sparse_matrix_t<ValueType>,
    elem::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<size_t,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution>,
        virtual public sketch_transform_t<base::sparse_matrix_t<ValueType>,
                                          elem::Matrix<ValueType> > {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef base::sparse_matrix_t<ValueType> matrix_type;
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
    hash_transform_t (int N, int S, skylark::base::context_t& context) :
        base_data_t (N, S, context) {}

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const base_data_t& other_data) :
        base_data_t(other_data) {}

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        columnwise_tag dimension) const {
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

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

private:

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) const {

        elem::Zero(sketch_of_A);

        double *SA = sketch_of_A.Buffer();
        int ld = sketch_of_A.LDim();

        const int* indptr = A.indptr();
        const int* indices = A.indices();
        const value_type* values = A.locked_values();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int col = 0; col < A.width(); col++) {
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];
                value_type val = values[j];
                SA[col * ld + base_data_t::row_idx[row]] +=
                    base_data_t::row_value[row] * val;
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

        double *SA = sketch_of_A.Buffer();
        int ld = sketch_of_A.LDim();

        const int* indptr = A.indptr();
        const int* indices = A.indices();
        const value_type* values = A.locked_values();

        for(int col = 0; col < A.width(); col++) {
#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for (int j = indptr[col]; j < indptr[col + 1]; j++) {
                int row = indices[j];
                value_type val = values[j];
                SA[base_data_t::row_idx[col] * ld + row] +=
                    base_data_t::row_value[col] * val;
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
                                     ValueDistribution>,
        virtual public sketch_transform_t<elem::DistMatrix<ValueType, ColDist,
                                                           RowDist>,
                                          elem::Matrix<ValueType> >  {

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
    hash_transform_t (int N, int S, skylark::base::context_t& context) :
        base_data_t (N, S, context) {}

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const base_data_t& other_data) :
        base_data_t(other_data) {}

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        columnwise_tag dimension) const {
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

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
        rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) const {

        // TODO this implementation is not communication efficient.
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

        // get communicator from matrix
        boost::mpi::communicator comm = skylark::utility::get_communicator(A);

        // Pull everything to rank-0
        boost::mpi::reduce (comm,
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

        // TODO this implementation is not communication efficient.
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

        // get communicator from matrix
        boost::mpi::communicator comm = skylark::utility::get_communicator(A);

        // Pull everything to rank-0
        boost::mpi::reduce (comm,
            SA_part.LockedBuffer(),
            SA_part.MemorySize(),
            sketch_of_A.Buffer(),
            std::plus<value_type>(),
            0);
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
