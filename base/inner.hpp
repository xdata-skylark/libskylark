#ifndef SKYLARK_INNER_HPP
#define SKYLARK_INNER_HPP

#include <boost/mpi.hpp>

#if SKYLARK_HAVE_ELEMENTAL

namespace skylark { namespace base {

template<typename T>
inline elem::Base<T> Nrm2(const elem::Matrix<T>& x) {
    return elem::Nrm2(x);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T>& x) {
    return elem::Nrm2(x);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T, elem::VC, elem::STAR>& x) {
    boost::mpi::communicator comm(x.DistComm(), boost::mpi::comm_attach);
    T local = elem::Nrm2(x.LockedMatrix());
    T snrm = boost::mpi::all_reduce(comm, local * local, std::plus<T>());
    return sqrt(snrm);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T, elem::VR, elem::STAR>& x) {
    boost::mpi::communicator comm(x.DistComm(), boost::mpi::comm_attach);
    T local = elem::Nrm2(x.LockedMatrix());
    T snrm = boost::mpi::all_reduce(comm, local * local, std::plus<T>());
    return sqrt(snrm);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T, elem::STAR, elem::STAR>& x) {
    return elem::Nrm2(x.LockedMatrix());
}

template<typename T>
inline void ColumnNrm2(const elem::Matrix<T>& A,
    elem::Matrix<elem::Base<T> >& N) {

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.Height(); i++)
            n[j] += a[j * A.LDim() + i] * elem::Conj(a[j * A.LDim() + i]);
        n[j] = sqrt(n[j]);
    }
}

template<typename T>
inline void ColumnNrm2(const elem::DistMatrix<T, elem::STAR, elem::STAR>& A,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& N) {

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.LocalHeight(); i++)
            n[j] += a[j * A.LDim() + i] * elem::Conj(a[j * A.LDim() + i]);
        n[j] = sqrt(n[j]);
    }
}

template<typename T>
inline void ColumnNrm2(const elem::DistMatrix<T, elem::VC, elem::STAR>& A,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& N) {

    std::vector<T> n(A.Width(), 1);
    const elem::Matrix<T> &Al = A.LockedMatrix();
    const double *a = Al.LockedBuffer();
    for(int j = 0; j < Al.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < Al.Height(); i++)
            n[j] += a[j * Al.LDim() + i] * elem::Conj(a[j * Al.LDim() + i]);
    }
    N.Resize(A.Width(), 1);
    elem::Zero(N);
    boost::mpi::communicator comm(N.Grid().Comm(), boost::mpi::comm_attach);
    boost::mpi::all_reduce(comm, n.data(), A.Width(), N.Buffer(), std::plus<T>());
    for(int j = 0; j < A.Width(); j++)
        N.Set(j, 0, sqrt(N.Get(j, 0)));
}

template<typename T>
inline void ColumnDot(const elem::Matrix<T>& A, const elem::Matrix<T>& B,
    elem::Matrix<elem::Base<T> >& N) {

    // TODO just assuming sizes are OK for now.

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    const double *b = B.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.Height(); i++)
            n[j] += a[j * A.LDim() + i] * elem::Conj(b[j * B.LDim() + i]);
    }
}

template<typename T>
inline void ColumnDot(const elem::DistMatrix<T, elem::STAR, elem::STAR>& A,
    const elem::DistMatrix<T, elem::STAR, elem::STAR>& B,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& N) {

    // TODO just assuming sizes are OK for now.

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    const double *b = B.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.LocalHeight(); i++)
            n[j] += a[j * A.LDim() + i] * elem::Conj(b[j * B.LDim() + i]);
    }
}

template<typename T>
inline void ColumnDot(const elem::DistMatrix<T, elem::VC, elem::STAR>& A,
    const elem::DistMatrix<T, elem::VC, elem::STAR>& B,
    elem::DistMatrix<elem::Base<T>, elem::STAR, elem::STAR>& N) {

    // TODO just assuming sizes are OK for now.

    std::vector<T> n(A.Width(), 1);
    const elem::Matrix<T> &Al = A.LockedMatrix();
    const double *a = Al.LockedBuffer();
    const elem::Matrix<T> &Bl = B.LockedMatrix();
    const double *b = Bl.LockedBuffer(); 
   for(int j = 0; j < Al.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < Al.Height(); i++)
            n[j] += a[j * Al.LDim() + i] * elem::Conj(b[j * Bl.LDim() + i]);
    }
    N.Resize(A.Width(), 1);
    elem::Zero(N);
    boost::mpi::communicator comm(N.Grid().Comm(), boost::mpi::comm_attach);
    boost::mpi::all_reduce(comm, n.data(), A.Width(), N.Buffer(), std::plus<T>());
}

template<typename T>
inline void RowDot(const elem::Matrix<T>& A, const elem::Matrix<T>& B,
    elem::Matrix<elem::Base<T> >& N) {

    // TODO just assuming sizes are OK for now.

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    const double *b = B.LockedBuffer();
    for(int i = 0; i < A.Height(); i++)
        n[i] = 0.0;

    for(int j = 0; j < A.Width(); j++) {
        for(int i = 0; i < A.Height(); i++)
            n[i] += a[j * A.LDim() + i] * elem::Conj(b[j * B.LDim() + i]);
    }
}

} } // namespace skylark::base

#endif // SKYLARK_HAVE_ELEMENTAL

#endif // SKYLARK_INNER_HPP
