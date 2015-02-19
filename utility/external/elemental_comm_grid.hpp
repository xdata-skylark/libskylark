#ifndef SKYLARK_ELEMENTAL_COMM_GRID_HPP
#define SKYLARK_ELEMENTAL_COMM_GRID_HPP

// A collection of helpers to compute information about distribution of
// Elemental objects.

namespace skylark {
namespace utility {

#include <El.hpp>

/// computes the rank owning the global (row_idx, col_idx) element
//template<typename index_type, typename value_type,
         //El::Distribution ColDist,
         //El::Distribution RowDist>
//index_type owner(const El::DistMatrix<value_type, ColDist, RowDist> &A,
                 //const index_type row_idx, const index_type col_idx) {

    //return A.Owner(row_idx, col_idx);
//}

/// computes the rank owning the global (row_idx, col_idx) Elent
template<typename index_type, typename value_type>
index_type owner(const El::DistMatrix<value_type> &A,
                 const index_type row_idx, const index_type col_idx) {

    return (col_idx % A.Grid().Width()) * A.Grid().Height() +
           (row_idx % A.Grid().Height());
}

template<typename index_type, typename value_type>
index_type owner(const El::DistMatrix<value_type, El::STAR, El::VR> &A,
                 const index_type row_idx, const index_type col_idx) {

    return (col_idx / A.Grid().Width()) % A.Grid().Height() +
           (col_idx % A.Grid().Width()) * A.Grid().Height();
}

template<typename index_type, typename value_type>
index_type owner(const El::DistMatrix<value_type, El::VR, El::STAR> &A,
                 const index_type row_idx, const index_type col_idx) {

    return (row_idx / A.Grid().Width()) % A.Grid().Height() +
           (row_idx % A.Grid().Width()) * A.Grid().Height();
}

template<typename index_type, typename value_type>
index_type owner(const El::DistMatrix<value_type, El::STAR, El::VC> &A,
                 const index_type row_idx, const index_type col_idx) {

    return col_idx % (A.Grid().Height() * A.Grid().Width());
}

template<typename index_type, typename value_type>
index_type owner(const El::DistMatrix<value_type, El::VC, El::STAR> &A,
                 const index_type row_idx, const index_type col_idx) {

    return row_idx % (A.Grid().Height() * A.Grid().Width());
}

} } //namespace skylark::utility

#endif //SKYLARK_ELENTAL_COMM_GRID_HPP
