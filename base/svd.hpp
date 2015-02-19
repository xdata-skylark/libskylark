#ifndef SKYLARK_SVD_HPP
#define SKYLARK_SVD_HPP

#include <El.hpp>

#include "Gemm.hpp"

namespace skylark { namespace base {

template<typename T>
void SVD(El::Matrix<T>& A, El::Matrix< El::Base<T> >& S,
    El::Matrix<T>& V) {
    El::SVD(A, S, V);
}

template<typename T>
void SVD(const El::Matrix<T>& A, El::Matrix<T>& U,
    El::Matrix<El::Base<T> >& S,
    El::Matrix<T>& V) {

    U = A;
    El::SVD(U, S, V);
}

template<typename T>
void SVD(El::DistMatrix<T, El::STAR, El::STAR>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& S,
    El::DistMatrix<T, El::STAR, El::STAR>& V) {
    El::SVD(A.Matrix(), S.Matrix(), V.Matrix());
    A.Resize(A.Matrix().Height(), A.Matrix().Width());
    S.Resize(S.Matrix().Height(), S.Matrix().Width());
    V.Resize(V.Matrix().Height(), V.Matrix().Width());
}

template<typename T>
void SVD(const El::DistMatrix<T, El::STAR, El::STAR>& A,
    El::DistMatrix<T, El::STAR, El::STAR>& U,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& S,
    El::DistMatrix<T, El::STAR, El::STAR>& V) {

    U = A;
    El::SVD(U.Matrix(), S.Matrix(), V.Matrix());
    U.Resize(U.Matrix().Height(), U.Matrix().Width());
    S.Resize(S.Matrix().Height(), S.Matrix().Width());
    V.Resize(V.Matrix().Height(), V.Matrix().Width());
}

template<typename T>
void SVD(El::DistMatrix<T>& A,
    El::DistMatrix<El::Base<T>, El::VR, El::STAR>& S,
    El::DistMatrix<T>& V) {
    El::SVD(A, S, V);
}

template<typename T>
void SVD(const El::DistMatrix<T>& A, El::DistMatrix<T>& U,
    El::DistMatrix<El::Base<T>, El::VR, El::STAR>& S,
    El::DistMatrix<T>& V) {

    U = A;
    El::SVD(U, S, V);
}

template<typename T>
void SVD(El::DistMatrix<T, El::VC,    El::STAR>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& S,
    El::DistMatrix<T, El::STAR, El::STAR>& V) {

    El::DistMatrix<T, El::VC,   El::STAR> Q;
    El::DistMatrix<T, El::STAR, El::STAR> R;
    El::Matrix<T> U_tilda;

    // tall and skinny QR (TSQR)
    Q = A;
    El::qr::ExplicitTS(Q, R);

    // local SVD of R
    base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    A.Resize(Q.Height(), S.Height());
    base::Gemm(El::NORMAL, El::NORMAL, T(1), Q, R, T(0), A);
}

template<typename T>
void SVD(const El::DistMatrix<T, El::VC,    El::STAR>& A,
    El::DistMatrix<T, El::VC,    El::STAR>& U,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& S,
    El::DistMatrix<T, El::STAR, El::STAR>& V) {

    El::DistMatrix<T, El::VC,   El::STAR> Q;
    El::DistMatrix<T, El::STAR, El::STAR> R;

    // tall and skinny QR (TSQR)
    Q = A;
    El::qr::ExplicitTS(Q, R);

    // local SVD of R
     base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    U.Resize(Q.Height(), S.Height());
    base::Gemm(El::NORMAL, El::NORMAL, T(1), Q, R, T(0), U);
}

template<typename T>
void SVD(El::DistMatrix<T, El::VR,    El::STAR>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& S,
    El::DistMatrix<T, El::STAR, El::STAR>& V) {

    El::DistMatrix<T, El::VR,   El::STAR> Q;
    El::DistMatrix<T, El::STAR, El::STAR> R;

    // tall and skinny QR (TSQR)
    Q = A;
    El::qr::ExplicitTS(A, R);

    // local SVD of R
    base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    A.Resize(A.Height(), S.Height());
    base::Gemm(El::NORMAL, El::NORMAL, T(1), Q, R, T(0), A);
}

template<typename T>
void SVD(const El::DistMatrix<T, El::VR,    El::STAR>& A,
    El::DistMatrix<T, El::VR,    El::STAR>& U,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& S,
    El::DistMatrix<T, El::STAR, El::STAR>& V) {

    El::DistMatrix<T, El::VR,   El::STAR> Q;
    El::DistMatrix<T, El::STAR, El::STAR> R;

    // tall and skinny QR (TSQR)
    Q = A;
    El::qr::ExplicitTS(Q, R);

    // local SVD of R
    base::SVD(R, S, V);
    S.Resize(S.Height(), 1);
    V.Resize(V.Height(), V.Width());

    // Compute U
    U.Resize(Q.Height(), S.Height());
    base::Gemm(El::NORMAL, El::NORMAL, T(1), Q, R, T(0), U);
}

template<typename T>
void SVD(const El::DistMatrix<T, El::STAR, El::VC>& A,
               El::DistMatrix<T, El::STAR, El::STAR>& U,
               El::DistMatrix<T, El::STAR, El::STAR>& S,
               El::DistMatrix<T, El::VC,   El::STAR>& V) {

    El::DistMatrix<T, El::VC, El::STAR> A_U_STAR;
    El::Adjoint(A, A_U_STAR);
    SVD(A_U_STAR, V, S, U);
}


template<typename T>
void SVD(const El::DistMatrix<T, El::STAR, El::VR>& A,
               El::DistMatrix<T, El::STAR, El::STAR>& U,
               El::DistMatrix<T, El::STAR, El::STAR>& S,
               El::DistMatrix<T, El::VR,   El::STAR>& V) {

    El::DistMatrix<T, El::VR, El::STAR> A_U_STAR;
    El::Adjoint(A, A_U_STAR);
    SVD(A_U_STAR, V, S, U);
}


} } // namespace skylark::base

#endif // SKYLARK_SVD_HPP
