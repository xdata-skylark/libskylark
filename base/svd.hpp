#ifndef SKYLARK_SVD_HPP
#define SKYLARK_SVD_HPP

#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif


namespace skylark { namespace base {

#if SKYLARK_HAVE_ELEMENTAL


template<typename T>
void svd(const elem::Matrix<T>& A,
         elem::Matrix<T>& U,
         elem::Matrix<T>& S,
         elem::Matrix<T>& V) {

    U = A;
    elem::SVD(U, S, V);
}


template<typename T>
void svd(const elem::DistMatrix<T>& A,
         elem::DistMatrix<T>& U,
         elem::DistMatrix<T, elem::VR, elem::STAR>& S,
         elem::DistMatrix<T>& V) {

    U = A;
    elem::SVD(U, S, V);
}


template<typename T>
void svd(const elem::DistMatrix<T, elem::VC,    elem::STAR>& A,
               elem::DistMatrix<T, elem::VC,    elem::STAR>& U,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& S,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    elem::DistMatrix<T, elem::VC,   elem::STAR> Q;
    elem::DistMatrix<T, elem::STAR, elem::STAR> R;
    elem::Matrix<T> U_tilda;

    // tall and skinny QR (TSQR)
    Q = A;
    elem::qr::ExplicitTS(Q, R);
    U_tilda = R.Matrix();

    // local SVD of R
    elem::SVD(U_tilda, S.Matrix(), V.Matrix());
    S.Resize(S.Matrix().Height(), 1);
    V.Resize(V.Matrix().Height(), V.Matrix().Width());

    // Compute U
    U.Resize(Q.Height(), S.Matrix().Height());
    elem::Gemm(elem::NORMAL, elem::NORMAL,
        T(1), Q.Matrix(), U_tilda, T(0), U.Matrix());
}


template<typename T>
void svd(const elem::DistMatrix<T, elem::VR,   elem::STAR>& A,
               elem::DistMatrix<T, elem::VR,   elem::STAR>& U,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& S,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    elem::DistMatrix<T, elem::VR,   elem::STAR> Q;
    elem::DistMatrix<T, elem::STAR, elem::STAR> R;
    elem::Matrix<T> U_tilda;

    // tall and skinny QR (TSQR)
    Q = A;
    elem::qr::ExplicitTS(Q, R);
    U_tilda = R.Matrix();

    // local SVD of R
    elem::SVD(U_tilda, S.Matrix(), V.Matrix());
    S.Resize(S.Matrix().Height(), 1);
    V.Resize(V.Matrix().Height(), V.Matrix().Width());

    // Compute U
    U.Resize(Q.Height(), S.Matrix().Height());
    elem::Gemm(elem::NORMAL, elem::NORMAL,
        T(1), Q.Matrix(), U_tilda, T(0), U.Matrix());
}


template<typename T>
void svd(const elem::DistMatrix<T, elem::STAR, elem::VC>& A,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& U,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& S,
               elem::DistMatrix<T, elem::VC,   elem::STAR>& V) {

    elem::DistMatrix<T, elem::VC, elem::STAR> A_U_STAR;
    elem::Adjoint(A, A_U_STAR);
    svd(A_U_STAR, V, S, U);
}


template<typename T>
void svd(const elem::DistMatrix<T, elem::STAR, elem::VR>& A,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& U,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& S,
               elem::DistMatrix<T, elem::VR,   elem::STAR>& V) {

    elem::DistMatrix<T, elem::VR, elem::STAR> A_U_STAR;
    elem::Adjoint(A, A_U_STAR);
    svd(A_U_STAR, V, S, U);
}


#endif // SKYLARK_HAVE_ELEMENTAL

} } // namespace skylark::base

#endif // SKYLARK_SVD_HPP
