#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_LOCAL_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_LOCAL_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL
#include "sketch_params.hpp"
#include "dense_transform_Elemental_coldist_star.hpp"
#endif

namespace skylark { namespace sketch {
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

        switch(ColDist) {
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

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_LOCAL

    /**
     * High-performance implementations
     */

    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::rowwise_tag tag) const {

        typedef elem::DistMatrix<value_type, elem::CIRC, elem::CIRC>
            intermediate_matrix_type;

        matrix_type sketch_of_A_CD_STAR(data_type::_S,
                             data_type::_N);
        intermediate_matrix_type sketch_of_A_CIRC_CIRC(data_type::_S,
                             data_type::_N);

        dense_transform_t<matrix_type, matrix_type, ValueDistribution>
            transform(data_type::_N, data_type::_S, data_type::_context);

        transform.apply(A, sketch_of_A_CD_STAR, tag);

        sketch_of_A_CIRC_CIRC = sketch_of_A_CD_STAR;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }
    }


    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::columnwise_tag tag) const {

        typedef elem::DistMatrix<value_type, elem::CIRC, elem::CIRC>
            intermediate_matrix_type;

        matrix_type sketch_of_A_CD_STAR(data_type::_S,
                             data_type::_N);
        intermediate_matrix_type sketch_of_A_CIRC_CIRC(data_type::_S,
                             data_type::_N);

        dense_transform_t<matrix_type, matrix_type, ValueDistribution>
            transform(data_type::_N, data_type::_S, data_type::_context);

        transform.apply(A, sketch_of_A_CD_STAR, tag);

        sketch_of_A_CIRC_CIRC = sketch_of_A_CD_STAR;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }
    }

////////////////////////////////////////////////////////////////////////////////

#else // HP_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_LOCAL

////////////////////////////////////////////////////////////////////////////////

    /**
     * BASE implementations
     */

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           columnwise_tag) const {

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

        // Pull everything to rank-0
        boost::mpi::reduce (utility::get_communicator(A),
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
                          rowwise_tag) const {

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

        // Collect at rank 0.
        boost::mpi::communicator comm = utility::get_communicator(A);
        utility::collect_dist_matrix(comm, comm.rank() == 0,
            SA_dist, sketch_of_A);
    }

#endif

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_LOCAL_HPP
