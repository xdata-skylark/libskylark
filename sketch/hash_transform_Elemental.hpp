#ifndef SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
#define SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP

#include "../utility/get_communicator.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization local input, local output
 */
template <typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    El::Matrix<ValueType>,
    El::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef El::Matrix<value_type> matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;
    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<IdxDistributionType,
                                  ValueDistribution> data_type;
    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const data_type& other_data) :
        data_type(other_data) {}

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
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) const {

        El::Zero(sketch_of_A);

        // Construct Pi * A (directly on the fly)
        for (size_t row_idx = 0; row_idx < A.Height(); row_idx++) {

            size_t new_row_idx      = data_type::row_idx[row_idx];
            value_type scale_factor = data_type::row_value[row_idx];

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

        El::Zero(sketch_of_A);

        // Construct Pi * A (directly on the fly)
        for (size_t col_idx = 0; col_idx < A.Width(); col_idx++) {

            size_t new_col_idx      = data_type::row_idx[col_idx];
            value_type scale_factor = data_type::row_value[col_idx];

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
    El::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef base::sparse_matrix_t<ValueType> matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;
    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<IdxDistributionType,
                                  ValueDistribution> data_type;
    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const data_type& other_data) :
        data_type(other_data) {}

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
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl (const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag) const {

        El::Zero(sketch_of_A);

        value_type *SA = sketch_of_A.Buffer();
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
                SA[col * ld + data_type::row_idx[row]] +=
                    data_type::row_value[row] * val;
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

        El::Zero(sketch_of_A);

        value_type *SA = sketch_of_A.Buffer();
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
                SA[data_type::row_idx[col] * ld + row] +=
                    data_type::row_value[col] * val;
            }
        }

    }
};

/**
 * Specialization: [Whatever, Whatever] -> [CIRC, CIRC]
 */
template <typename ValueType,
          El::Distribution ColDist,
          El::Distribution RowDist,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    El::DistMatrix<ValueType, ColDist, RowDist>,
    El::DistMatrix<ValueType, El::CIRC, El::CIRC>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, ColDist, RowDist> matrix_type;
    typedef El::DistMatrix<value_type, El::CIRC, El::CIRC> output_matrix_type;
    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<IdxDistributionType,
                                  ValueDistribution> data_type;
    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const data_type& other_data) :
        data_type(other_data) {}

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
                base::elemental_exception()
                    << base::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
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

        // TODO this implementation is not communication efficient.
        // Sketching a nxd matrix to sxd will communicate O(sdP)
        // doubles, when you can sometime communicate less:
        // For [MC,MR] or [MR,MC] you need O(sd sqrt(P)).
        // For [*, VC/VR] you need only O(sd).

        // Create space to hold local part of SA
        El::Matrix<value_type> SA_part((El::Int)(this->_S), A.Width(),
            (El::Int)(this->_S));

        // Construct Pi * A (directly on the fly)
        El::Zero(SA_part);
        for (size_t j = 0; j < A.LocalHeight(); j++) {

            size_t row_idx = A.ColShift() + A.ColStride() * j;
            size_t new_row_idx      = data_type::row_idx[row_idx];
            value_type scale_factor = data_type::row_value[row_idx];

            for(size_t i = 0; i < A.LocalWidth(); i++) {
                size_t col_idx = A.RowShift() + A.RowStride() * i;
                value_type value = scale_factor * A.GetLocal(j, i);
                SA_part.Update(new_row_idx, col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::communicator comm = utility::get_communicator(A);
        boost::mpi::reduce(comm,
            SA_part.LockedBuffer(),
            SA_part.MemorySize(),
            sketch_of_A.Buffer(),
            std::plus<value_type>(),
            0);

        if(comm.rank() == 0 && sketch_of_A.LDim() != this->_S) {
            value_type *buf = sketch_of_A.Buffer();
            for(size_t i = A.Width() - 1; i > 0; i--)
                memcpy(buf + sketch_of_A.LDim() * i, buf + i * this->_S,
                    this->_S * sizeof(value_type));
        }
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
        El::Matrix<value_type> SA_part(A.Height(), this->_S, A.Height());

        // Construct A * Pi (directly on the fly)
        El::Zero(SA_part);
        for (size_t j = 0; j < A.LocalWidth(); ++j) {

            size_t col_idx = A.RowShift() + A.RowStride() * j;
            size_t new_col_idx = data_type::row_idx[col_idx];
            value_type scale_factor = data_type::row_value[col_idx];

            for(size_t i = 0; i < A.LocalHeight(); ++i) {
                size_t row_idx   = A.ColShift() + A.ColStride() * i;
                value_type value = scale_factor *  A.GetLocal(i, j);
                SA_part.Update(row_idx, new_col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::communicator comm = utility::get_communicator(A);
        boost::mpi::reduce (comm,
            SA_part.LockedBuffer(),
            SA_part.MemorySize(),
            sketch_of_A.Buffer(),
            std::plus<value_type>(),
            0);

        if(comm.rank() == 0 && sketch_of_A.LDim() != A.Height()) {
            value_type *buf = sketch_of_A.Buffer();
            for(size_t i = this->_S - 1; i > 0; i--)
                memcpy(buf + sketch_of_A.LDim() * i, buf + i * A.Height(),
                    A.Height() * sizeof(value_type));
        }
    }
};

/**
 * Specialization: [Whatever, Whatever] -> [STAR, STAR]
 */
template <typename ValueType,
          El::Distribution ColDist,
          El::Distribution RowDist,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    El::DistMatrix<ValueType, ColDist, RowDist>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution>,
        virtual public sketch_transform_t<El::DistMatrix<ValueType, ColDist,
                                                           RowDist>,
                                          El::DistMatrix<ValueType,
                                                           El::STAR,
                                                           El::STAR> >  {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, ColDist, RowDist> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> output_matrix_type;
    typedef IdxDistributionType<size_t> idx_distribution_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef hash_transform_data_t<IdxDistributionType,
                                  ValueDistribution> data_type;
    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    hash_transform_t (const hash_transform_t<matrix_type,
                                       output_matrix_type,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (const data_type& other_data) :
        data_type(other_data) {}

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
                base::elemental_exception()
                    << base::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
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
        El::Matrix<value_type> SA_part (sketch_of_A.Height(),
            sketch_of_A.Width(),
            sketch_of_A.LDim());

        El::Zero(SA_part);

        // Construct Pi * A (directly on the fly)
        for (size_t j = 0; j < A.LocalHeight(); j++) {

            size_t row_idx = A.ColShift() + A.ColStride() * j;
            size_t new_row_idx      = data_type::row_idx[row_idx];
            value_type scale_factor = data_type::row_value[row_idx];

            for(size_t i = 0; i < A.LocalWidth(); i++) {
                size_t col_idx = A.RowShift() + A.RowStride() * i;
                value_type value = scale_factor * A.GetLocal(j, i);
                SA_part.Update(new_row_idx, col_idx, value);
            }
        }

        boost::mpi::all_reduce (utility::get_communicator(A),
                            SA_part.LockedBuffer(),
                            SA_part.MemorySize(),
                            sketch_of_A.Buffer(),
                            std::plus<value_type>());
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
        El::Matrix<value_type> SA_part (sketch_of_A.Height(),
            sketch_of_A.Width(),
            sketch_of_A.LDim());

        El::Zero(SA_part);

        // Construct A * Pi (directly on the fly)
        for (size_t j = 0; j < A.LocalWidth(); ++j) {

            size_t col_idx = A.RowShift() + A.RowStride() * j;
            size_t new_col_idx = data_type::row_idx[col_idx];
            value_type scale_factor = data_type::row_value[col_idx];

            for(size_t i = 0; i < A.LocalHeight(); ++i) {
                size_t row_idx   = A.ColShift() + A.ColStride() * i;
                value_type value = scale_factor *  A.GetLocal(i, j);
                SA_part.Update(row_idx, new_col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::all_reduce (utility::get_communicator(A),
                            SA_part.LockedBuffer(),
                            SA_part.MemorySize(),
                            sketch_of_A.Buffer(),
                            std::plus<value_type>());
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_ELEMENTAL_HPP
