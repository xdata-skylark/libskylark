#ifndef SKYLARK_COMBBLAS_COMM_GRID_HPP
#define SKYLARK_COMBBLAS_COMM_GRID_HPP

// A collection of helpers to compute information about distribution of
// CombBLAS objects.

namespace skylark {
namespace utility {

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>

/// computes the global start row offset for rank
template<typename index_type, typename value_type>
inline index_type cb_row_offset(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
    const size_t rank) {

    return static_cast<size_t>((static_cast<double>(A.getnrow())
        / A.getcommgrid()->GetGridRows()))
        * A.getcommgrid()->GetRankInProcCol(rank);
}

/// computes the global start row offset for the calling rank
template<typename index_type, typename value_type>
inline index_type cb_my_row_offset(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A) {

    return cb_row_offset(A, A.getcommgrid()->GetRank());
}

/// computes the global start column offset for rank
template<typename index_type, typename value_type>
inline index_type cb_col_offset(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
    const size_t rank) {

    return static_cast<size_t>((static_cast<double>(A.getncol())
        / A.getcommgrid()->GetGridCols()))
        * A.getcommgrid()->GetRankInProcRow(rank);
}

/// computes the global start column offset for the calling rank
template<typename index_type, typename value_type>
inline index_type cb_my_col_offset(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A) {

    return cb_col_offset(A, A.getcommgrid()->GetRank());
}

/// computes the number of rows per processor
template<typename index_type, typename value_type>
index_type cb_rows_per_proc(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A) {

    const size_t grows = A.getcommgrid()->GetGridRows();
    return static_cast<size_t>(A.getnrow() / grows);
}

/// computes the number of columns per processor
template<typename index_type, typename value_type>
index_type cb_cols_per_proc(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A) {

    const size_t gcols = A.getcommgrid()->GetGridCols();
    return static_cast<size_t>(A.getncol() / gcols);
}

/// computes the rank owning the global (row_idx, col_idx) element
template<typename index_type, typename value_type>
index_type compute_target_rank(
    const SpParMat<index_type, value_type, SpDCCols<index_type, value_type> > &A,
    const size_t row_idx, const size_t col_idx) {

    //FIXME: functor member variables
    const size_t grows = A.getcommgrid()->GetGridRows();
    const size_t rows_per_proc = static_cast<size_t>(A.getnrow() / grows);
    const size_t gcols = A.getcommgrid()->GetGridCols();
    const size_t cols_per_proc = static_cast<size_t>(A.getncol() / gcols);

    return A.getcommgrid()->GetRank(
        std::min(static_cast<size_t>(row_idx / rows_per_proc), grows - 1),
        std::min(static_cast<size_t>(col_idx / cols_per_proc), gcols - 1));
}

#endif //HAVE_COMBBLAS

} } //namespace skylark::utility

#endif //SKYLARK_COMBBLAS_COMM_GRID_HPP
