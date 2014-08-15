#ifndef SKYLARK_ELEMENTAL_COMM_GRID_HPP
#define SKYLARK_ELEMENTAL_COMM_GRID_HPP

// A collection of helpers to compute information about distribution of
// Elemental objects.

namespace skylark {
namespace utility {

#include <elemental.hpp>

/// computes the rank owning the global (row_idx, col_idx) element
//template<typename index_type, typename value_type,
         //elem::Distribution ColDist,
         //elem::Distribution RowDist>
//index_type owner(const elem::DistMatrix<value_type, ColDist, RowDist> &A,
                 //const index_type row_idx, const index_type col_idx) {

    //return A.Owner(row_idx, col_idx);
//}

/// computes the rank owning the global (row_idx, col_idx) element
template<typename index_type, typename value_type>
index_type owner(const elem::DistMatrix<value_type> &A,
                 const index_type row_idx, const index_type col_idx) {

    return (col_idx % A.Grid().Width()) * A.Grid().Height() +
           (row_idx % A.Grid().Height());
}

template<typename index_type, typename value_type>
index_type owner(const elem::DistMatrix<value_type, elem::STAR, elem::VR> &A,
                 const index_type row_idx, const index_type col_idx) {

    return (col_idx / A.Grid().Width()) % A.Grid().Height() +
           (col_idx % A.Grid().Width()) * A.Grid().Height();
}

template<typename index_type, typename value_type>
index_type owner(const elem::DistMatrix<value_type, elem::VR, elem::STAR> &A,
                 const index_type row_idx, const index_type col_idx) {

    return (row_idx / A.Grid().Width()) % A.Grid().Height() +
           (row_idx % A.Grid().Width()) * A.Grid().Height();
}

template<typename index_type, typename value_type>
index_type owner(const elem::DistMatrix<value_type, elem::STAR, elem::VC> &A,
                 const index_type row_idx, const index_type col_idx) {

    return col_idx % (A.Grid().Height() * A.Grid().Width());
}

template<typename index_type, typename value_type>
index_type owner(const elem::DistMatrix<value_type, elem::VC, elem::STAR> &A,
                 const index_type row_idx, const index_type col_idx) {

    return row_idx % (A.Grid().Height() * A.Grid().Width());
}

} } //namespace skylark::utility

#endif //SKYLARK_ELEMENTAL_COMM_GRID_HPP
