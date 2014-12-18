#ifndef SKYLARK_QR_HPP
#define SKYLARK_QR_HPP

#include <El.hpp>


namespace skylark { namespace base { namespace qr {

// TODO not sure we need this file anymore

template<typename T>
void ExplicitUnitary(El::Matrix<T>& A) {

    El::qr::ExplicitUnitary(A);
}

template<typename T>
void ExplicitUnitary(El::DistMatrix<T>& A) {

    El::qr::ExplicitUnitary(A);
}

template<typename T>
void ExplicitUnitary(El::DistMatrix<T, El::VC, El::STAR>& A) {

    El::DistMatrix<T, El::STAR, El::STAR> R;
    El::qr::ExplicitTS(A, R);
}


template<typename T>
void ExplicitUnitary(El::DistMatrix<T, El::VR, El::STAR>& A) {

    El::DistMatrix<T, El::STAR, El::STAR> R;
    El::qr::ExplicitTS(A, R);
}

} } } // namespace skylark::base::qr

#endif // SKYLARK_QR_HPP
