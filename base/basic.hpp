#ifndef SKYLARK_SCALE_HPP
#define SKYLARK_SCALE_HPP

namespace skylark { namespace base {


#if SKYLARK_HAVE_ELEMENTAL

template<typename T>
inline void Zero(elem::Matrix<T>& A) {
    elem::Zero(A);
}

template<typename T, elem::Distribution U, elem::Distribution V>
inline void Zero(elem::DistMatrix<T, U, V>& A) {
    elem::Zero(A);
}


template<typename T>
inline void DiagonalScale(elem::LeftOrRight s, elem::Orientation o, 
    const elem::Matrix<T>& d, elem::Matrix<T>& X) {
    elem::DiagonalScale(s, o, d, X);
}

template<typename T, elem::Distribution U, elem::Distribution V, 
         elem::Distribution W, elem::Distribution Z>
inline void DiagonalScale(elem::LeftOrRight s, elem::Orientation o, 
    const elem::DistMatrix<T, U, V>& d, elem::DistMatrix<T, W, Z>& X) {
    elem::DiagonalScale(s, o, d, X);
}

template<typename T>
inline void Axpy(T alpha, const elem::Matrix<T>& X, elem::Matrix<T>& Y) {
    elem::Axpy(alpha, X, Y);
}

template<typename T,
         elem::Distribution U1, elem::Distribution V1,
         elem::Distribution U2, elem::Distribution V2>
inline void Axpy(T alpha, const elem::DistMatrix<T, U1, V1>& X,
    elem::DistMatrix<T, U2, V2>& Y) {
    elem::Axpy(alpha, X, Y);
}

/**
 * Axpy between the column vectors in the matrices. Allows different scalar
 *  values applied to different columns of X.
 */
template<typename T>
inline void Axpy(const elem::Matrix<T>& alphas,
    const elem::Matrix<T>& X, elem::Matrix<T>& Y) {
    elem::Matrix<T> Xv, Yv;
    for(int i = 0; i < X.Width(); i++) {
        elem::View(Yv, Y, 0, i, Y.Height(), 1);
        elem::LockedView(Xv, X, 0, i, Y.Height(), 1);
        elem::Axpy(alphas.Get(i, 0), Xv, Yv);
    }
}

template<typename T,
         elem::Distribution U1, elem::Distribution V1,
         elem::Distribution U2, elem::Distribution V2>
inline void Axpy(const elem::DistMatrix<T, elem::STAR, elem::STAR> &alphas,
    const elem::DistMatrix<T, U1, V1>& X,
    elem::DistMatrix<T, U2, V2>& Y) {
    elem::DistMatrix<T, U1, V1> Xv;
    elem::DistMatrix<T, U2, V2> Yv;
    for(int i = 0; i < X.Width(); i++) {
        elem::View(Yv, Y, 0, i, Y.Height(), 1);
        elem::LockedView(Xv, X, 0, i, Y.Height(), 1);
        elem::Axpy(alphas.Get(i, 0), Xv, Yv);
    }
}


#endif

} } // namespace skylark::base

#endif // SKYLARK_COPY_HPP
