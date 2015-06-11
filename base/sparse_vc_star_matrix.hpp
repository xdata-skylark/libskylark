#ifndef SKYLARK_SPARSE_VC_STAR_MATRIX_HPP
#define SKYLARK_SPARSE_VC_STAR_MATRIX_HPP

#include <map>
#include <memory>
#include <vector>

#include <El.hpp>

#include "sparse_dist_matrix.hpp"

namespace skylark { namespace base {

/**
 *  This implements a very crude sparse VC / STAR matrix using a CSC sparse
 *  matrix container intended to hold the local sparse matrix.
 */
template<typename ValueType=double>
struct sparse_vc_star_matrix_t : public sparse_dist_matrix_t<ValueType> {

    typedef sparse_dist_matrix_t<ValueType> base_t;

    sparse_vc_star_matrix_t()
        : base_t() {

        //FIXME
    }

    sparse_vc_star_matrix_t(
            El::Int n_rows, El::Int n_cols, boost::mpi::communicator& comm,
            const El::Grid& grid)
        : base_t(n_rows, n_cols, comm, grid) {

        base_t::_row_align = 0;
        base_t::_col_align = 0;

        base_t::_row_stride = 1;
        base_t::_col_stride = grid.VCSize();

        base_t::_col_shift = grid.VCRank();
        base_t::_row_shift = 0;

        base_t::_col_rank = El::mpi::Rank(grid.VCComm());
        base_t::_row_rank = El::mpi::Rank(El::mpi::COMM_SELF);
    }

};

} }

#endif
