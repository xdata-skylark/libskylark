#ifndef SKYLARK_TRSM_HPP
#define SKYLARK_TRSM_HPP

// Defines a generic Trsm function that recieves a wider set of matrices

namespace skylark { namespace base {

template<typename T>
inline void Trsm(El::LeftOrRight s, El::UpperOrLower ul,
    El::Orientation oA, El::UnitOrNonUnit diag, T alpha,
    const El::Matrix<T>& A, El::Matrix<T>& B) {
    El::Trsm(s, ul, oA, diag, alpha, A, B);
}

template<typename T>
inline void Trsm(El::LeftOrRight s, El::UpperOrLower ul,
    El::Orientation oA, El::UnitOrNonUnit diag, T alpha,
    const El::DistMatrix<T, El::STAR, El::STAR>& A,
    El::DistMatrix<T, El::STAR, El::STAR>& B) {
    El::Trsm(s, ul, oA, diag, alpha, A.LockedMatrix(), B.Matrix());
}

template<typename T>
inline void Trsm(El::LeftOrRight s, El::UpperOrLower ul,
    El::Orientation oA, El::UnitOrNonUnit diag, T alpha,
    const El::DistMatrix<T, El::CIRC, El::CIRC>& A,
    El::DistMatrix<T, El::CIRC, El::CIRC>& B) {
    // TODO: check passing A, B with same grid.
    if (B.Grid().Rank() == 0)
        El::Trsm(s, ul, oA, diag, alpha, A.LockedMatrix(), B.Matrix());
}

template<typename T>
inline void Trsm(El::LeftOrRight s, El::UpperOrLower ul,
    El::Orientation oA, El::UnitOrNonUnit diag, T alpha,
    const El::DistMatrix<T>& A, El::DistMatrix<T>& B) {
    El::Trsm(s, ul, oA, diag, alpha, A, B);
}

} } // namespace skylark::base

#endif // SKYLARK_TRSM_HPP
