#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_LOCAL_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_LOCAL_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL
#include "sketch_params.hpp"
#include "dense_transform_Elemental_mc_mr.hpp"
#endif

namespace skylark { namespace sketch {
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

private:

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL

    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         skylark::sketch::rowwise_tag tag) const {

        typedef elem::DistMatrix<value_type, elem::CIRC, elem::CIRC>
            intermediate_matrix_type;

        matrix_type sketch_of_A_MC_MR(base_data_t::S,
                             base_data_t::N);
        intermediate_matrix_type sketch_of_A_CIRC_CIRC(base_data_t::S,
                             base_data_t::N);

        dense_transform_t<matrix_type, matrix_type> transform(base_data_t::N,
            base_data_t::S, base_data_t::context);

        transform.apply(A, sketch_of_A_MC_MR, tag);

        sketch_of_A_CIRC_CIRC = sketch_of_A_MC_MR;

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
                         skylark::sketch::colwise_tag tag) const {

        typedef elem::DistMatrix<value_type, elem::CIRC, elem::CIRC>
            intermediate_matrix_type;

        matrix_type sketch_of_A_MC_MR(base_data_t::S,
                             base_data_t::N);
        intermediate_matrix_type sketch_of_A_CIRC_CIRC(base_data_t::S,
                             base_data_t::N);

        dense_transform_t<matrix_type, matrix_type> transform(base_data_t::N,
            base_data_t::S, base_data_t::context);

        transform.apply(A, sketch_of_A_MC_MR, tag);

        sketch_of_A_CIRC_CIRC = sketch_of_A_MC_MR;

        boost::mpi::communicator world;
        MPI_Comm mpi_world(world);
        elem::Grid grid(mpi_world);
        int rank = world.rank();
        if (rank == 0) {
            sketch_of_A = sketch_of_A_CIRC_CIRC.Matrix();
        }
    }


#else

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for distributed input [MC, MR], local output
     * and columnwise.
     */
    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          columnwise_tag) const {
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
                         rowwise_tag) const {
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

#endif

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_LOCAL_HPP
