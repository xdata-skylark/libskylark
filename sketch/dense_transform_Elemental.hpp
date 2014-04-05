#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/exception.hpp"
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
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {}

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
                utility::elemental_exception()
                    << utility::error_msg(e.what()) );
        }
    }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for local and columnwise.
     */
    void apply_impl_local(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

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
                          skylark::sketch::rowwise_tag) const {

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

/**
 * Specialization distributed input, local output, for [SOMETHING, *]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType>,
    ValueDistribution > :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                  ValueDistribution> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {}

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
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) const {

        // Create space to hold partial SA --- for 1D, we need SA space
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
                                          sketch_of_A.Width(),
                                          sketch_of_A.LDim());
        elem::Zero(SA_part);

        // To avoid allocating a huge S_local matrix we are breaking
        // S_local into column slices, and multiply one by one.
        // The number of columns in each slice is A's width
        // since that way the slice take the same amount of memory as
        // the sketch.

        int slice_width = A.Width();


        elem::Matrix<value_type> S_local(data_type::_S, slice_width);
        for (int js = 0; js < A.LocalHeight(); js += slice_width) {
            int je = std::min(js + slice_width, A.LocalHeight());
            // adapt size of local portion (can be less than slice_width)
            S_local.Resize(data_type::_S, je-js);
            for(int j = js; j < je; j++) {
                int col = A.ColShift() + A.ColStride() * j;
                for (int i = 0; i < data_type::_S; i++) {
                    value_type sample =
                        data_type::random_samples[col * data_type::_S + i];
                    S_local.Set(i, j-js, data_type::scale * sample);
                }
            }

            elem::Matrix<value_type> A_slice;
            elem::LockedView(A_slice, A.LockedMatrix(),
                js, 0, je-js, A.Width());

            // Do the multiplication
            base::Gemm (elem::NORMAL,
                elem::NORMAL,
                1.0,
                S_local,
                A_slice,
                1.0,
                SA_part);
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
      * Implementation for [VR/VC, *] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        // Create a distributed matrix to hold the output.
        //  We later gather to a dense matrix.
        matrix_type SA_dist(A.Height(), data_type::_S, A.Grid());

        // Create S. Since it is rowwise, we assume it can be held in memory.
        elem::Matrix<value_type> S_local(data_type::_S, data_type::_N);
        for (int j = 0; j < data_type::_N; j++) {
            for (int i = 0; i < data_type::_S; i++) {
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S_local.Set(i, j, data_type::scale * sample);
            }
        }

        // Apply S to the local part of A to get the local part of SA.
        base::Gemm(elem::NORMAL,
            elem::TRANSPOSE,
            1.0,
            A.LockedMatrix(),
            S_local,
            0.0,
            SA_dist.Matrix());

        // get communicator from matrix
        boost::mpi::communicator comm = skylark::utility::get_communicator(A);
        int rank = comm.rank();

        // Collect at rank 0.
        // TODO Grid rank 0 or context rank 0?
        skylark::utility::collect_dist_matrix(comm,
            rank == 0,
            SA_dist, sketch_of_A);
    }

};


/**
 * Specialization distributed input and output in [SOMETHING, *]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR>
    output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                   ValueDistribution> data_type;

    /**
     * Regular Constructor
     */
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {}

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
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) const {

        // Redistribute matrix A: [VC/VR, STAR] -> [STAR, VC/VR]
        elem::DistMatrix<value_type, elem::STAR, ColDist> A_STAR_ColDist(A);

        elem::DistMatrix<value_type,
                         elem::STAR, ColDist>
            sketch_of_A_STAR_ColDist(sketch_of_A.Height(), sketch_of_A.Width());
        elem::Zero(sketch_of_A_STAR_ColDist);

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
            elem::View(sketch_slice, sketch_of_A_STAR_ColDist.Matrix(),
                S_num_rows_consumed, 0,
                S_part_height, A_STAR_ColDist.LocalWidth());
            // Do the multiplication: S_part*A
            base::Gemm (elem::NORMAL,
                elem::NORMAL,
                1.0,
                S_part,
                A_STAR_ColDist.LockedMatrix(),
                0.0,
                sketch_slice);
            S_num_rows_consumed += S_part_height;
        }
        // Redistribute the sketch: [STAR, VC/VR] -> [VC/VR, STAR]
        sketch_of_A = sketch_of_A_STAR_ColDist;
    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [VR/VC, *] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {


        // Create S. Since it is rowwise, we assume it can be held in memory.
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
            A.LockedMatrix(),
            S_local,
            0.0,
            sketch_of_A.Matrix());
    }

};


/**
 * Specialization distributed input, local output, for [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution RowDist,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    elem::Matrix<ValueType>,
    ValueDistribution > :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                  ValueDistribution> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {}

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
     * Implementation for [*, VR/VC] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) const {

        elem::DistMatrix<value_type,
                         elem::STAR, RowDist>
            sketch_of_A_STAR_RowDist(sketch_of_A.Height(), sketch_of_A.Width());
        elem::Zero(sketch_of_A_STAR_RowDist);

        elem::DistMatrix<value_type,
                         elem::CIRC,
                         elem::CIRC> sketch_of_A_CIRC_CIRC(data_type::_S,
                             data_type::_N);

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
        sketch_of_A_CIRC_CIRC = sketch_of_A_STAR_RowDist;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }

    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [*, VR/VC] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        // Redistribute matrix A: [STAR, VC/VR] -> [VC/VR, STAR]
        elem::DistMatrix<value_type, RowDist, elem::STAR> A_RowDist_STAR(A);

        elem::DistMatrix<value_type,
                         RowDist,
                         elem::STAR>
            sketch_of_A_RowDist_STAR(sketch_of_A.Height(), sketch_of_A.Width());
        elem::Zero(sketch_of_A_RowDist_STAR);

        elem::DistMatrix<value_type,
                         elem::CIRC,
                         elem::CIRC> sketch_of_A_CIRC_CIRC(data_type::_S,
                             data_type::_N);

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
            0.0,
            sketch_of_A_RowDist_STAR.Matrix());

        sketch_of_A_CIRC_CIRC = sketch_of_A_RowDist_STAR;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }

    }

};

/**
 * Specialization distributed input and output in [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution RowDist,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> matrix_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist>
    output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                   ValueDistribution> data_type;

    /**
     * Regular Constructor
     */
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {}

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
     * Implementation for [*, VR/VC] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) const {

        elem::Zero(sketch_of_A);

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
            elem::View(sketch_slice, sketch_of_A.Matrix(),
                S_num_rows_consumed, 0,
                S_part_height, sketch_of_A.LocalWidth());
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
    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [*, VR/VC] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

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
            0.0,
            sketch_of_A.Matrix());

    // Redistribute the sketch: [VC/VR, STAR] -> [STAR, VC/VR]
    sketch_of_A = sketch_of_A_RowDist_STAR;
    }
};


/**
 * Specialization distributed input [MC, MR], local output
 */
template <typename ValueType,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType>,
    elem::Matrix<ValueType>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                   ValueDistribution> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {}

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
        try {
            apply_impl_dist(A, sketch_of_A, dimension);
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
     * Implementation for distributed input [MC, MR], local output
     * and columnwise.
     */
    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {
        elem::DistMatrix<value_type> S(data_type::_S, data_type::_N);
        elem::DistMatrix<value_type,
                         elem::MC,
                         elem::MR> sketch_of_A_MC_MR(data_type::_S,
                             data_type::_N);
        elem::DistMatrix<value_type,
                         elem::CIRC,
                         elem::CIRC> sketch_of_A_CIRC_CIRC(data_type::_S,
                             data_type::_N);

        for(int j_loc = 0; j_loc < S.LocalWidth(); ++j_loc) {
            int j = S.RowShift() + S.RowStride() * j_loc;
            for (int i_loc = 0; i_loc < S.LocalHeight(); ++i_loc) {
                int i = S.ColShift() + S.ColStride() * i_loc;
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.SetLocal(i_loc, j_loc, data_type::scale * sample);
            }
        }

        base::Gemm (elem::NORMAL,
                    elem::NORMAL,
                    1.0,
                    S,
                    A,
                    0.0,
                    sketch_of_A_MC_MR);
        sketch_of_A_CIRC_CIRC = sketch_of_A_MC_MR;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for distributed input [MC, MR], local output
     * and rowwise.
     */
    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::rowwise_tag) const {
        elem::DistMatrix<value_type> S(data_type::_S, data_type::_N);
        elem::DistMatrix<value_type,
                         elem::MC,
                         elem::MR> sketch_of_A_MC_MR(data_type::_S,
                             data_type::_N);
        elem::DistMatrix<value_type,
                         elem::CIRC,
                         elem::CIRC> sketch_of_A_CIRC_CIRC(data_type::_S,
                             data_type::_N);

        for(int j_loc = 0; j_loc < S.LocalWidth(); ++j_loc) {
            int j = S.RowShift() + S.RowStride() * j_loc;
            for (int i_loc = 0; i_loc < S.LocalHeight(); ++i_loc) {
                int i = S.ColShift() + S.ColStride() * i_loc;
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.SetLocal(i_loc, j_loc, data_type::scale * sample);
            }
        }

        base::Gemm (elem::NORMAL,
                    elem::TRANSPOSE,
                    1.0,
                    A,
                    S,
                    0.0,
                    sketch_of_A_MC_MR);
        sketch_of_A_CIRC_CIRC = sketch_of_A_MC_MR;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }

    }
};



/**
 * Specialization distributed input [MC, MR], distributed output [MC, MR]
 */
template <typename ValueType,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType>,
    elem::DistMatrix<ValueType>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type> matrix_type;
    typedef elem::DistMatrix<value_type> output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                   ValueDistribution> data_type;

    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, skylark::base::context_t& context)
        : data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValueDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    dense_transform_t(const data_type& other_data)
        : data_type(other_data) {

    }

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
     * Implementation for distributed input/output [MC, MR] and columnwise.
     */
    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        elem::DistMatrix<value_type> S(data_type::_S, data_type::_N);

        for(int j_loc = 0; j_loc < S.LocalWidth(); ++j_loc) {
            int j = S.RowShift() + S.RowStride() * j_loc;
            for (int i_loc = 0; i_loc < S.LocalHeight(); ++i_loc) {
                int i = S.ColShift() + S.ColStride() * i_loc;
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.SetLocal(i_loc, j_loc, data_type::scale * sample);
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
     * Implementation for distributed input/output [MC, MR] and rowwise.
     */
    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::rowwise_tag) const {

        elem::DistMatrix<value_type> S(data_type::_S, data_type::_N);

        for(int j_loc = 0; j_loc < S.LocalWidth(); ++j_loc) {
            int j = S.RowShift() + S.RowStride() * j_loc;
            for (int i_loc = 0; i_loc < S.LocalHeight(); ++i_loc) {
                int i = S.ColShift() + S.ColStride() * i_loc;
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.SetLocal(i_loc, j_loc, data_type::scale * sample);
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

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_HPP
