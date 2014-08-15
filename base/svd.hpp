#ifndef SKYLARK_SVD_HPP
#define SKYLARK_SVD_HPP

#include <elemental.hpp>

#include "Gemm.hpp"

namespace skylark { namespace base {

template<typename T>
void SVD(elem::Matrix<T>& A, elem::Matrix< elem::Base<T> >& S,
    elem::Matrix<T>& V) {
    elem::SVD(A, S, V);
}

template<typename T>
void SVD(const elem::Matrix<T>& A, elem::Matrix<T>& U,
    elem::Matrix<elem::Base<T> >& S,
    elem::Matrix<T>& V) {

    U = A;
    elem::SVD(U, S, V);
}

template<typename T>
void SVD(elem::DistMatrix<T, elem::STAR, elem::STAR>& A,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& S,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {
    elem::SVD(A.Matrix(), S.Matrix(), V.Matrix());
    A.Resize(A.Matrix().Height(), A.Matrix().Width());
    S.Resize(S.Matrix().Height(), S.Matrix().Width());
    V.Resize(V.Matrix().Height(), V.Matrix().Width());
}

template<typename T>
void SVD(const elem::DistMatrix<T, elem::STAR, elem::STAR>& A,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& U,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& S,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    U = A;
    elem::SVD(U.Matrix(), S.Matrix(), V.Matrix());
    U.Resize(U.Matrix().Height(), U.Matrix().Width());
    S.Resize(S.Matrix().Height(), S.Matrix().Width());
    V.Resize(V.Matrix().Height(), V.Matrix().Width());
}

template<typename T>
void SVD(elem::DistMatrix<T>& A,
    elem::DistMatrix<elem::Base<T>, elem::VR, elem::STAR>& S,
    elem::DistMatrix<T>& V) {
    elem::SVD(A, S, V);
}

template<typename T>
void SVD(const elem::DistMatrix<T>& A, elem::DistMatrix<T>& U,
    elem::DistMatrix<elem::Base<T>, elem::VR, elem::STAR>& S,
    elem::DistMatrix<T>& V) {

    U = A;
    elem::SVD(U, S, V);
}

template<typename T>
void SVD(elem::DistMatrix<T, elem::VC,    elem::STAR>& A,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& S,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    elem::DistMatrix<T, elem::VC,   elem::STAR> Q;
    elem::DistMatrix<T, elem::STAR, elem::STAR> R;
    elem::Matrix<T> U_tilda;

    // tall and skinny QR (TSQR)
    Q = A;
    elem::qr::ExplicitTS(Q, R);

    // local SVD of R
    base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    A.Resize(Q.Height(), S.Height());
    base::Gemm(elem::NORMAL, elem::NORMAL, T(1), Q, R, T(0), A);
}

template<typename T>
void SVD(const elem::DistMatrix<T, elem::VC,    elem::STAR>& A,
    elem::DistMatrix<T, elem::VC,    elem::STAR>& U,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& S,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    elem::DistMatrix<T, elem::VC,   elem::STAR> Q;
    elem::DistMatrix<T, elem::STAR, elem::STAR> R;

    // tall and skinny QR (TSQR)
    Q = A;
    elem::qr::ExplicitTS(Q, R);

    // local SVD of R
     base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    U.Resize(Q.Height(), S.Height());
    base::Gemm(elem::NORMAL, elem::NORMAL, T(1), Q, R, T(0), U);
}

template<typename T>
void SVD(elem::DistMatrix<T, elem::VR,    elem::STAR>& A,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& S,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    elem::DistMatrix<T, elem::VR,   elem::STAR> Q;
    elem::DistMatrix<T, elem::STAR, elem::STAR> R;

    // tall and skinny QR (TSQR)
    Q = A;
    elem::qr::ExplicitTS(A, R);

    // local SVD of R
    base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    A.Resize(A.Height(), S.Height());
    base::Gemm(elem::NORMAL, elem::NORMAL, T(1), Q, R, T(0), A);
}

template<typename T>
void SVD(const elem::DistMatrix<T, elem::VR,    elem::STAR>& A,
    elem::DistMatrix<T, elem::VR,    elem::STAR>& U,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& S,
    elem::DistMatrix<T, elem::STAR, elem::STAR>& V) {

    elem::DistMatrix<T, elem::VR,   elem::STAR> Q;
    elem::DistMatrix<T, elem::STAR, elem::STAR> R;

    // tall and skinny QR (TSQR)
    Q = A;
    elem::qr::ExplicitTS(Q, R);

    // local SVD of R
    base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    U.Resize(Q.Height(), S.Height());
    base::Gemm(elem::NORMAL, elem::NORMAL, T(1), Q, R, T(0), U);
}

template<typename T>
void SVD(const elem::DistMatrix<T, elem::STAR, elem::VC>& A,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& U,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& S,
               elem::DistMatrix<T, elem::VC,   elem::STAR>& V) {

    elem::DistMatrix<T, elem::VC, elem::STAR> A_U_STAR;
    elem::Adjoint(A, A_U_STAR);
    SVD(A_U_STAR, V, S, U);
}


template<typename T>
void SVD(const elem::DistMatrix<T, elem::STAR, elem::VR>& A,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& U,
               elem::DistMatrix<T, elem::STAR, elem::STAR>& S,
               elem::DistMatrix<T, elem::VR,   elem::STAR>& V) {

    elem::DistMatrix<T, elem::VR, elem::STAR> A_U_STAR;
    elem::Adjoint(A, A_U_STAR);
    SVD(A_U_STAR, V, S, U);
}


} } // namespace skylark::base

#endif // SKYLARK_SVD_HPP
