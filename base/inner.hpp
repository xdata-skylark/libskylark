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

    N.Resize(A.Width(), 1);
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
    N.Resize(A.Width(), 1);
    double *n = N.Buffer();
    const double *a = A.LockedBuffer();
    for(int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(int i = 0; i < A.LocalHeight(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(a[j * A.LDim() + i]);
        n[j] = sqrt(n[j]);
    }
}

template<typename T, El::Distribution U, El::Distribution V>
inline void ColumnNrm2(const El::DistMatrix<T, U, V>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {

    std::vector<T> n(A.Width(), 1);
    std::fill(n.begin(), n.end(), 0.0);
    const El::Matrix<T> &Al = A.LockedMatrix();
    const double *a = Al.LockedBuffer();
    for(int j = 0; j < Al.Width(); j++)
        for(int i = 0; i < Al.Height(); i++)
            n[A.GlobalCol(j)] +=
                a[j * Al.LDim() + i] * El::Conj(a[j * Al.LDim() + i]);

    N.Resize(A.Width(), 1);
    El::Zero(N);
    boost::mpi::communicator comm(N.Grid().Comm().comm, boost::mpi::comm_attach);
    boost::mpi::all_reduce(comm, n.data(), A.Width(), N.Buffer(), std::plus<T>());
    for(int j = 0; j < A.Width(); j++)
        N.Set(j, 0, sqrt(N.Get(j, 0)));
}

template<typename T>
inline void ColumnNrm2(const El::AbstractDistMatrix<T>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {

    N.Resize(A.Width(), 1);

    if (A.Participating()) {
        std::vector<T> n(A.Width(), 1);
        std::fill(n.begin(), n.end(), 0.0);
        const El::Matrix<T> &Al = A.LockedMatrix();
        const double *a = Al.LockedBuffer();
        for(int j = 0; j < Al.Width(); j++)
            for(int i = 0; i < Al.Height(); i++)
                n[A.GlobalCol(j)] +=
                    a[j * Al.LDim() + i] * El::Conj(a[j * Al.LDim() + i]);


        El::Zero(N);
        El::mpi::AllReduce(n.data(), N.Buffer(), A.Width(), MPI_SUM,
            A.DistComm());
        for(int j = 0; j < A.Width(); j++)
            N.Set(j, 0, sqrt(N.Get(j, 0)));
    }

    El::mpi::Broadcast(N.Buffer(), A.Width(), A.Root(), A.CrossComm());
}


template<typename T>
inline void ColumnDot(const El::Matrix<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& N) {

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
    El::DistMatrix<T, El::STAR, El::STAR>& N) {

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

template<typename T, El::Distribution U, El::Distribution V>
inline void ColumnDot(const El::DistMatrix<T, U, V>& A,
    const El::DistMatrix<T, U, V>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& N) {

    // TODO just assuming sizes are OK for now, and grid aligned.
    std::vector<T> n(A.Width(), 1);
    std::fill(n.begin(), n.end(), 0);
    const El::Matrix<T> &Al = A.LockedMatrix();
    const double *a = Al.LockedBuffer();
    const El::Matrix<T> &Bl = B.LockedMatrix();
    const double *b = Bl.LockedBuffer();
    for(int j = 0; j < Al.Width(); j++)
        for(int i = 0; i < Al.Height(); i++)
            n[A.GlobalCol(j)] +=
                a[j * Al.LDim() + i] * El::Conj(b[j * Bl.LDim() + i]);

   N.Resize(A.Width(), 1);
   El::Zero(N);
   boost::mpi::communicator comm(N.Grid().Comm().comm, boost::mpi::comm_attach);
   boost::mpi::all_reduce(comm, n.data(), A.Width(), N.Buffer(), std::plus<T>());
}

template<typename T>
inline void RowDot(const El::Matrix<T>& A, const El::Matrix<T>& B, 
    El::Matrix<T>& N) {

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

/**
 * C = beta * C + alpha * square_euclidean_distance_matrix(A, B)
 */
template<typename T>
void Euclidean(direction_t dirA, direction_t dirB, T alpha,
    const El::Matrix<T> &A, const El::Matrix<T> &B,
    T beta, El::Matrix<T> &C) {

    T *c = C.Buffer();
    El::Int ldC = C.LDim();

    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        base::Gemm(El::ADJOINT, El::NORMAL, -2.0 * alpha, A, B, beta, C);

        El::Matrix<T> NA, NB;
        ColumnNrm2(A, NA);
        ColumnNrm2(B, NB);
        T *na = NA.Buffer(), *nb = NB.Buffer();

        El::Int m = base::Width(A);
        El::Int n = base::Width(B);

        for(El::Int j = 0; j < n; j++)
            for(El::Int i = 0; i < m; i++)
                c[j * ldC + i] += alpha * (na[i] * na[i] + nb[j] * nb[j]);

    }

    // TODO the rest of the cases.
}

template<typename T>
void Euclidean(direction_t dirA, direction_t dirB, T alpha,
    const El::AbstractDistMatrix<T> &A, const El::AbstractDistMatrix<T> &B,
    T beta, El::AbstractDistMatrix<T> &C) {

    T *c = C.Buffer();
    El::Int ldC = C.LDim();

    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        El::Gemm(El::ADJOINT, El::NORMAL, -2.0 * alpha, A, B, beta, C);

        El::DistMatrix<T, El::STAR, El::STAR> NA, NB;
        ColumnNrm2(A, NA);
        ColumnNrm2(B, NB);
        T *na = NA.Buffer(), *nb = NB.Buffer();

        El::Int m = C.LocalHeight();
        El::Int n = C.LocalWidth();

        for(El::Int j = 0; j < n; j++)
            for(El::Int i = 0; i < m; i++) {
                T a = na[C.GlobalRow(i)];
                T b = nb[C.GlobalCol(j)];
                c[j * ldC + i] += alpha * (a * a + b * b);
            }
    }

    // TODO the rest of the cases.
}

/**
 * C = beta * C + alpha * square_euclidean_distance_matrix(A, A)
 * Update only lower part.
 */
template<typename T>
void SymmetricEuclidean(El::UpperOrLower uplo, direction_t dir, T alpha,
    const El::Matrix<T> &A, T beta, El::Matrix<T> &C) {

    T *c = C.Buffer();
    int ldC = C.LDim();

    if (dir == base::COLUMNS) {
        El::Herk(uplo, El::ADJOINT, -2.0 * alpha, A, beta, C);

        El::Matrix<T> N;
        ColumnNrm2(A, N);
        T *nn = N.Buffer();;

        int n = base::Width(A);

        for(El::Int j = 0; j < n; j++)
            for(El::Int i = ((uplo == El::UPPER) ? 0 : j);
                i < ((uplo == El::UPPER) ? j : n); i++)
                c[j * ldC + i] += alpha * (nn[i] * nn[i] + nn[j] * nn[j]);

    }

    // TODO the rest of the cases.
}

template<typename T>
void SymmetricEuclidean(El::UpperOrLower uplo, direction_t dir, T alpha,
    const El::AbstractDistMatrix<T> &A, T beta, El::AbstractDistMatrix<T> &C) {

    T *c = C.Buffer();
    int ldC = C.LDim();

    if (dir == base::COLUMNS) {
        El::Herk(uplo, El::ADJOINT, -2.0 * alpha, A, beta, C);

        El::DistMatrix<El::Base<T>, El::STAR, El::STAR > N;
        ColumnNrm2(A, N);
        El::Base<T> *nn = N.Buffer();;

        int n = C.LocalWidth();
        int m = C.LocalHeight();

        for(int j = 0; j < n; j++)
            for(El::Int i =
                    ((uplo == El::UPPER) ? 0 : C.LocalRowOffset(A.GlobalCol(j)));
                i < ((uplo == El::UPPER) ? C.LocalRowOffset(A.GlobalCol(j) + 1) : m); i++) {
                El::Base<T> a = nn[C.GlobalRow(i)];
                El::Base<T> b = nn[C.GlobalCol(j)];
                c[j * ldC + i] += alpha * (a * a + b * b);
            }
    }

    // TODO the rest of the cases.
}


} } // namespace skylark::base

#endif // SKYLARK_INNER_HPP
