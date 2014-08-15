#ifndef SKYLARK_QR_HPP
#define SKYLARK_QR_HPP

#include <elemental.hpp>


namespace skylark { namespace base { namespace qr {

template<typename T>
void Explicit(elem::Matrix<T>& A, bool colPiv=false) {

    elem::qr::Explicit(A, colPiv);
}


template<typename T>
void Explicit(elem::DistMatrix<T>& A, bool colPiv=false) {

    elem::qr::Explicit(A, colPiv);
}


template<typename T>
void Explicit(elem::DistMatrix<T, elem::VC, elem::STAR>& A) {

    elem::DistMatrix<T, elem::STAR, elem::STAR> R;
    elem::qr::ExplicitTS(A, R);
}


template<typename T>
void Explicit(elem::DistMatrix<T, elem::VR, elem::STAR>& A) {

    elem::DistMatrix<T, elem::STAR, elem::STAR> R;
    elem::qr::ExplicitTS(A, R);
}

} } } // namespace skylark::base::qr

#endif // SKYLARK_QR_HPP
