#ifndef SKYLARK_SCALE_HPP
#define SKYLARK_SCALE_HPP

namespace skylark { namespace base {


template<typename T>
inline void Zero(El::Matrix<T>& A) {
    El::Zero(A);
}

template<typename T, El::Distribution U, El::Distribution V>
inline void Zero(El::DistMatrix<T, U, V>& A) {
    El::Zero(A);
}


template<typename T>
inline void DiagonalScale(El::LeftOrRight s, El::Orientation o,
    const El::Matrix<T>& d, El::Matrix<T>& X) {
    El::DiagonalScale(s, o, d, X);
}

template<typename T, El::Distribution U, El::Distribution V,
         El::Distribution W, El::Distribution Z>
inline void DiagonalScale(El::LeftOrRight s, El::Orientation o,
    const El::DistMatrix<T, U, V>& d, El::DistMatrix<T, W, Z>& X) {
    El::DiagonalScale(s, o, d, X);
}

template<typename T>
inline void Axpy(T alpha, const El::Matrix<T>& X, El::Matrix<T>& Y) {
    El::Axpy(alpha, X, Y);
}

template<typename T,
         El::Distribution U1, El::Distribution V1,
         El::Distribution U2, El::Distribution V2>
inline void Axpy(T alpha, const El::DistMatrix<T, U1, V1>& X,
    El::DistMatrix<T, U2, V2>& Y) {
    El::Axpy(alpha, X, Y);
}

/**
 * Axpy between the column vectors in the matrices. Allows different scalar
 *  values applied to different columns of X.
 */
template<typename T>
inline void Axpy(const El::Matrix<T>& alphas,
    const El::Matrix<T>& X, El::Matrix<T>& Y) {
    El::Matrix<T> Xv, Yv;
    for(int i = 0; i < X.Width(); i++) {
        El::View(Yv, Y, 0, i, Y.Height(), 1);
        El::LockedView(Xv, X, 0, i, Y.Height(), 1);
        El::Axpy(alphas.Get(i, 0), Xv, Yv);
    }
}

template<typename T,
         El::Distribution U1, El::Distribution V1,
         El::Distribution U2, El::Distribution V2>
inline void Axpy(const El::DistMatrix<T, El::STAR, El::STAR> &alphas,
    const El::DistMatrix<T, U1, V1>& X,
    El::DistMatrix<T, U2, V2>& Y) {
    El::DistMatrix<T, U1, V1> Xv;
    El::DistMatrix<T, U2, V2> Yv;
    for(int i = 0; i < X.Width(); i++) {
        El::View(Yv, Y, 0, i, Y.Height(), 1);
        El::LockedView(Xv, X, 0, i, Y.Height(), 1);
        El::Axpy(alphas.Get(i, 0), Xv, Yv);
    }
}

template<typename T, El::Distribution U, El::Distribution V>
inline void Scale(const T s, El::DistMatrix<T, U, V>& X) {
    El::Scale(s, X);
}


} } // namespace skylark::base

#endif // SKYLARK_COPY_HPP
