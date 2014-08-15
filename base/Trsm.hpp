#ifndef SKYLARK_TRSM_HPP
#define SKYLARK_TRSM_HPP

// Defines a generic Trsm function that recieves a wider set of matrices

namespace skylark { namespace base {

template<typename T>
inline void Trsm(elem::LeftOrRight s, elem::UpperOrLower ul,
    elem::Orientation oA, elem::UnitOrNonUnit diag, T alpha,
    const elem::Matrix<T>& A, elem::Matrix<T>& B) {
    elem::Trsm(s, ul, oA, diag, alpha, A, B);
}

template<typename T>
inline void Trsm(elem::LeftOrRight s, elem::UpperOrLower ul,
    elem::Orientation oA, elem::UnitOrNonUnit diag, T alpha,
    const elem::DistMatrix<T, elem::STAR, elem::STAR>& A,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& B) {
    elem::Trsm(s, ul, oA, diag, alpha, A.LockedMatrix(), B.Matrix());
}

template<typename T>
inline void Trsm(elem::LeftOrRight s, elem::UpperOrLower ul,
    elem::Orientation oA, elem::UnitOrNonUnit diag, T alpha,
    const elem::DistMatrix<T, elem::CIRC, elem::CIRC>& A,
    elem::DistMatrix<T, elem::CIRC, elem::CIRC>& B) {
    // TODO: check passing A, B with same grid.
    if (B.Grid().Rank() == 0)
        elem::Trsm(s, ul, oA, diag, alpha, A.LockedMatrix(), B.Matrix());
}

template<typename T>
inline void Trsm(elem::LeftOrRight s, elem::UpperOrLower ul,
    elem::Orientation oA, elem::UnitOrNonUnit diag, T alpha,
    const elem::DistMatrix<T>& A, elem::DistMatrix<T>& B) {
    elem::Trsm(s, ul, oA, diag, alpha, A, B);
}

} } // namespace skylark::base

#endif // SKYLARK_TRSM_HPP
