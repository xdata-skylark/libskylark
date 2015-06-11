#ifndef SKYLARK_SPARSE_STAR_VR_MATRIX_HPP
#define SKYLARK_SPARSE_STAR_VR_MATRIX_HPP

#include <El.hpp>

#include "sparse_dist_matrix.hpp"

namespace skylark { namespace base {

/**
 *  This implements a very crude sparse STAR / VR matrix using a CSC sparse
 *  matrix container intended to hold local sparse matrix.
 */
template<typename ValueType=double>
struct sparse_star_vr_matrix_t : public sparse_dist_matrix_t<ValueType> {

    typedef sparse_dist_matrix_t<ValueType> base_t;

    sparse_star_vr_matrix_t(
            El::Int n_rows, El::Int n_cols, boost::mpi::communicator& comm,
            const El::Grid& grid)
        : base_t(n_rows, n_cols, comm, grid) {

        base_t::_row_align = 0;
        base_t::_col_align = 0;

        base_t::_row_stride = grid.VRSize();
        base_t::_col_stride = 1;

        base_t::_col_shift = 0;
        base_t::_row_shift = grid.VRRank();

        base_t::_col_rank = El::mpi::Rank(El::mpi::COMM_SELF);
        base_t::_row_rank = El::mpi::Rank(grid.VRComm());
    }

};

} }

#endif
