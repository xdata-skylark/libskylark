#ifndef SKYLARK_INNER_HPP
#define SKYLARK_INNER_HPP

#include <boost/mpi.hpp>

namespace skylark { namespace base {

template<typename T>
inline El::Base<T> Nrm2(const El::Matrix<T>& x) {
    return El::Nrm2(x);
}

template<typename T>
inline El::Base<T> Nrm2(const El::DistMatrix<T>& x) {
    return El::Nrm2(x);
}

template<typename T>
inline El::Base<T> Nrm2(const El::DistMatrix<T, El::VC, El::STAR>& x) {
    boost::mpi::communicator comm(x.DistComm().comm, boost::mpi::comm_attach);
    T local = El::Nrm2(x.LockedMatrix());
    T snrm = boost::mpi::all_reduce(comm, local * local, std::plus<T>());
    return sqrt(snrm);
}

template<typename T>
inline El::Base<T> Nrm2(const El::DistMatrix<T, El::VR, El::STAR>& x) {
    boost::mpi::communicator comm(x.DistComm(), boost::mpi::comm_attach);
    T local = El::Nrm2(x.LockedMatrix());
    T snrm = boost::mpi::all_reduce(comm, local * local, std::plus<T>());
    return sqrt(snrm);
}

template<typename T>
inline El::Base<T> Nrm2(const El::DistMatrix<T, El::STAR, El::STAR>& x) {
    return El::Nrm2(x.LockedMatrix());
}

template<typename T>
inline void ColumnNrm2(const El::Matrix<T>& A,
    El::Matrix<El::Base<T> >& N) {

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.Height(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(a[j * A.LDim() + i]);
        n[j] = sqrt(n[j]);
    }
}

template<typename T>
inline void ColumnNrm2(const El::DistMatrix<T, El::STAR, El::STAR>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.LocalHeight(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(a[j * A.LDim() + i]);
        n[j] = sqrt(n[j]);
    }
}

template<typename T>
inline void ColumnNrm2(const El::DistMatrix<T, El::VC, El::STAR>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {

    std::vector<T> n(A.Width(), 1);
    const El::Matrix<T> &Al = A.LockedMatrix();
    const double *a = Al.LockedBuffer();
    for(int j = 0; j < Al.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < Al.Height(); i++)
            n[j] += a[j * Al.LDim() + i] * El::Conj(a[j * Al.LDim() + i]);
    }
    N.Resize(A.Width(), 1);
    El::Zero(N);
    boost::mpi::communicator comm(N.Grid().Comm().comm, boost::mpi::comm_attach);
    boost::mpi::all_reduce(comm, n.data(), A.Width(), N.Buffer(), std::plus<T>());
    for(int j = 0; j < A.Width(); j++)
        N.Set(j, 0, sqrt(N.Get(j, 0)));
}

template<typename T>
inline void ColumnDot(const El::Matrix<T>& A, const El::Matrix<T>& B,
    El::Matrix<El::Base<T> >& N) {

    // TODO just assuming sizes are OK for now.

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    const double *b = B.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.Height(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(b[j * B.LDim() + i]);
    }
}

template<typename T>
inline void ColumnDot(const El::DistMatrix<T, El::STAR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {

    // TODO just assuming sizes are OK for now.

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    const double *b = B.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.LocalHeight(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(b[j * B.LDim() + i]);
    }
}

template<typename T>
inline void ColumnDot(const El::DistMatrix<T, El::VC, El::STAR>& A,
    const El::DistMatrix<T, El::VC, El::STAR>& B,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {

    // TODO just assuming sizes are OK for now.

    std::vector<T> n(A.Width(), 1);
    const El::Matrix<T> &Al = A.LockedMatrix();
    const double *a = Al.LockedBuffer();
    const El::Matrix<T> &Bl = B.LockedMatrix();
    const double *b = Bl.LockedBuffer();
   for(int j = 0; j < Al.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < Al.Height(); i++)
            n[j] += a[j * Al.LDim() + i] * El::Conj(b[j * Bl.LDim() + i]);
    }
    N.Resize(A.Width(), 1);
    El::Zero(N);
    boost::mpi::communicator comm(N.Grid().Comm(), boost::mpi::comm_attach);
    boost::mpi::all_reduce(comm, n.data(), A.Width(), N.Buffer(), std::plus<T>());
}

template<typename T>
inline void RowDot(const El::Matrix<T>& A, const El::Matrix<T>& B,
    El::Matrix<El::Base<T> >& N) {

    // TODO just assuming sizes are OK for now.

    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    const double *b = B.LockedBuffer();
    for(int i = 0; i < A.Height(); i++)
        n[i] = 0.0;

    for(int j = 0; j < A.Width(); j++) {
        for(int i = 0; i < A.Height(); i++)
            n[i] += a[j * A.LDim() + i] * El::Conj(b[j * B.LDim() + i]);
    }
}

} } // namespace skylark::base

#endif // SKYLARK_INNER_HPP
