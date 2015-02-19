#ifndef SKYLARK_GEMM_DETAIL_HPP
#define SKYLARK_GEMM_DETAIL_HPP

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#include <CommGrid.h>
#endif

#include "../utility/external/view.hpp"
#include "../utility/external/combblas_comm_grid.hpp"
#include "../utility/external/elemental_comm_grid.hpp"

namespace skylark { namespace base { namespace detail {

#if SKYLARK_HAVE_COMBBLAS

/// only compute local product:
///   elem(n x k) x local_part_cb(k x m) -> array(n x m)
template<typename index_type, typename value_type>
inline void mixed_gemm_local_part_nn (
        const double alpha,
        const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
        const El::DistMatrix<value_type, El::STAR, El::STAR> &S,
        const double beta,
        std::vector<value_type> &local_matrix) {

    typedef SpDCCols< index_type, value_type > col_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    matrix_type &_A = const_cast<matrix_type&>(A);
    col_t &data = _A.seq();

    //FIXME
    local_matrix.resize(S.Width() * data.getnrow(), 0);
    size_t cb_col_offset = utility::cb_my_col_offset(A);

    for(typename col_t::SpColIter col = data.begcol();
        col != data.endcol(); col++) {
        for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
            nz != data.endnz(col); nz++) {

            // we want local index here to fill local dense matrix
            index_type rowid = nz.rowid();
            // column needs to be global
            index_type colid = col.colid() + cb_col_offset;

            // compute application of S to yield a partial row in the result.
            for(size_t bcol = 0; bcol < S.Width(); ++bcol) {
                local_matrix[rowid * S.Width() + bcol] +=
                        alpha * S.Get(colid, bcol) * nz.value();
            }
        }
    }
}


/// implementing gemm for CB * (*/*) = (SOMETHING/*)
//FIXME: benchmark against one-sided
template<typename index_type, typename value_type, El::Distribution col_d>
inline void inner_panel_mixed_gemm_impl_nn(
        const double alpha,
        const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
        const El::DistMatrix<value_type, El::STAR, El::STAR> &S,
        const double beta,
        El::DistMatrix<value_type, col_d, El::STAR> &C) {

    int n_proc_side   = A.getcommgrid()->GetGridRows();
    int output_width  = S.Width();
    int output_height = A.getnrow();

    size_t rank = A.getcommgrid()->GetRank();
    size_t cb_row_offset = utility::cb_my_row_offset(A);

    typedef SpDCCols< index_type, value_type > col_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    matrix_type &_A = const_cast<matrix_type&>(A);
    col_t &data = _A.seq();

    // 1) compute the local values still using the CombBLAS distribution (2D
    //    processor grid). We assume the result is dense.
    std::vector<double> local_matrix;
    mixed_gemm_local_part_nn(alpha, A, S, 0.0, local_matrix);

    // 2) reduce first along rows so that each processor owns the values in
    //    the output row of the SOMETHING/* matrix and values for processors in
    //    the same processor column.
    boost::mpi::communicator my_row_comm(
            A.getcommgrid()->GetRowWorld(), boost::mpi::comm_duplicate);

    // storage for other procs in same row communicator: rank -> (row, values)
    typedef std::vector<std::pair<int, std::vector<double> > > for_rank_t;
    std::vector<for_rank_t> for_rank(n_proc_side);

    for(size_t local_row = 0; local_row < data.getnrow(); ++local_row) {

        size_t row = local_row + cb_row_offset;

        // the owner for VR/* and VC/* matrices is independent of the column
        size_t target_proc = utility::owner(C, row, static_cast<size_t>(0));

        // if the target processor is not in the current row communicator, get
        // the value in the processor grid sharing the same row.
        if(!A.getcommgrid()->OnSameProcRow(target_proc))
            target_proc = static_cast<int>(rank / n_proc_side) *
                            n_proc_side + target_proc % n_proc_side;

        size_t target_row_rank = A.getcommgrid()->GetRankInProcRow(target_proc);

        // reduce partial row (FIXME: if the resulting matrix is still
        // expected to be sparse, change this to communicate only nnz).
        // Working on local_width columns concurrently per column processing
        // group.
        size_t local_width = S.Width();
        const value_type* buffer = &local_matrix[local_row * local_width];
        std::vector<value_type> new_values(local_width);
        boost::mpi::reduce(my_row_comm, buffer, local_width,
                &new_values[0], std::plus<value_type>(), target_row_rank);

        // processor stores result directly if it is the owning rank of that
        // row, save for subsequent communication along rows otherwise
        if(rank == utility::owner(C, row, static_cast<size_t>(0))) {
            int elem_lrow = C.LocalRow(row);
            for(size_t idx = 0; idx < local_width; ++idx) {
                int elem_lcol = C.LocalCol(idx);
                C.SetLocal(elem_lrow, elem_lcol,
                    new_values[idx] + beta * C.GetLocal(elem_lrow, elem_lcol));
            }
        } else if (rank == target_proc) {
            // store for later comm across rows
            for_rank[utility::owner(C, row, static_cast<size_t>(0)) / n_proc_side].push_back(
                    std::make_pair(row, new_values));
        }
    }

    // 3) gather remaining values along rows: we exchange all the values with
    //    other processors in the same communicator row and then add them to
    //    our local part.
    boost::mpi::communicator my_col_comm(
            A.getcommgrid()->GetColWorld(), boost::mpi::comm_duplicate);

    std::vector<for_rank_t> new_values;
    for(int i = 0; i < n_proc_side; ++i)
        boost::mpi::gather(my_col_comm, for_rank[i], new_values, i);

    // insert new values
    for(size_t proc = 0; proc < new_values.size(); ++proc) {
        const for_rank_t &cur  = new_values[proc];

        for(size_t i = 0; i < cur.size(); ++i) {
            int elem_lrow = C.LocalRow(cur[i].first);
            for(size_t j = 0; j < cur[i].second.size(); ++j) {
                size_t elem_lcol = C.LocalCol(j);
                C.SetLocal(elem_lrow, elem_lcol,
                        cur[i].second[j] + beta *
                        C.GetLocal(elem_lrow, elem_lcol));
            }
        }
    }
}


//FIXME: benchmark against one-sided
template<typename index_type, typename value_type, El::Distribution col_d>
inline void outer_panel_mixed_gemm_impl_nn(
        const double alpha,
        const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
        const El::DistMatrix<value_type, El::STAR, El::STAR> &S,
        const double beta,
        El::DistMatrix<value_type, col_d, El::STAR> &C) {

    utility::combblas_slab_view_t<index_type, value_type> cbview(A, false);

    //FIXME: factor
    size_t slab_size = 2 * C.Grid().Height();
    for(size_t cur_row_idx = 0; cur_row_idx < cbview.nrows();
        cur_row_idx += slab_size) {

        size_t cur_slab_size =
            std::min(slab_size, cbview.nrows() - cur_row_idx);

        // get the next slab_size columns of B
        El::DistMatrix<value_type, col_d, El::STAR>
            A_row(cur_slab_size, S.Height());

        cbview.extract_elemental_row_slab_view(A_row, cur_slab_size);

        // assemble the distributed column vector
        for(size_t l_col_idx = 0; l_col_idx < A_row.LocalWidth();
            l_col_idx++) {

            size_t g_col_idx = l_col_idx * A_row.RowStride()
                               + A_row.RowShift();

            for(size_t l_row_idx = 0; l_row_idx < A_row.LocalHeight();
                ++l_row_idx) {

                size_t g_row_idx = l_row_idx * A_row.ColStride()
                                   + A_row.ColShift() + cur_row_idx;

                A_row.SetLocal(l_row_idx, l_col_idx,
                               cbview(g_row_idx, g_col_idx));
            }
        }

        El::DistMatrix<value_type, col_d, El::STAR>
            C_slice(cur_slab_size, C.Width());
        El::View(C_slice, C, cur_row_idx, 0, cur_slab_size, C.Width());
        El::LocalGemm(El::NORMAL, El::NORMAL, alpha, A_row, S,
                        beta, C_slice);
    }
}


//FIXME: benchmark against one-sided
template<typename index_type, typename value_type, El::Distribution col_d>
inline void outer_panel_mixed_gemm_impl_tn(
        const double alpha,
        const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
        const El::DistMatrix<value_type, col_d, El::STAR> &S,
        const double beta,
        El::DistMatrix<value_type, El::STAR, El::STAR> &C) {

    El::DistMatrix<value_type, El::STAR, El::STAR>
        tmp_C(C.Height(), C.Width());
    El::Zero(tmp_C);

    utility::combblas_slab_view_t<index_type, value_type> cbview(A, false);

    //FIXME: factor
    size_t slab_size = 2 * S.Grid().Height();
    for(size_t cur_row_idx = 0; cur_row_idx < cbview.ncols();
        cur_row_idx += slab_size) {

        size_t cur_slab_size =
            std::min(slab_size, cbview.ncols() - cur_row_idx);

        // get the next slab_size columns of B
        El::DistMatrix<value_type, El::STAR, El::STAR>
            A_row(cur_slab_size, S.Height());

        // transpose is column
        //cbview.extract_elemental_column_slab_view(A_row, cur_slab_size);
        cbview.extract_full_slab_view(cur_slab_size);

        // matrix mult (FIXME only iter nz)
        for(size_t l_row_idx = 0; l_row_idx < A_row.LocalHeight();
            ++l_row_idx) {

            size_t g_row_idx = l_row_idx * A_row.ColStride()
                               + A_row.ColShift() + cur_row_idx;

            for(size_t l_col_idx = 0; l_col_idx < A_row.LocalWidth();
                l_col_idx++) {

                //XXX: should be the same as l_col_idx
                size_t g_col_idx = l_col_idx * A_row.RowStride()
                                   + A_row.RowShift();

                // continue if we don't own values in S in this row
                if(!S.IsLocalRow(g_col_idx))
                    continue;

                //get transposed value
                value_type val = alpha * cbview(g_col_idx, g_row_idx);

                for(size_t s_col_idx = 0; s_col_idx < S.LocalWidth();
                    s_col_idx++) {

                    tmp_C.UpdateLocal(g_row_idx, s_col_idx,
                                val * S.GetLocal(S.LocalRow(g_col_idx), s_col_idx));
                }
            }
        }
    }

    //FIXME: scaling
    if(A.getcommgrid()->GetRank() == 0) {
        for(size_t col_idx = 0; col_idx < C.Width(); col_idx++)
            for(size_t row_idx = 0; row_idx < C.Height(); row_idx++)
                tmp_C.UpdateLocal(row_idx, col_idx,
                        beta * C.GetLocal(row_idx, col_idx));
    }

    //FIXME: Use utility getter
    boost::mpi::communicator world(
            A.getcommgrid()->GetWorld(), boost::mpi::comm_duplicate);
    boost::mpi::all_reduce (world,
                        tmp_C.LockedBuffer(),
                        C.Height() * C.Width(),
                        C.Buffer(),
                        std::plus<value_type>());
}

#endif // SKYLARK_HAVE_COMBBLAS

} } } // namespace skylark::base::detail

#endif //SKYLARK_GEMM_DETAIL_HPP
