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
    T *n = N.Buffer();
    const T *a = A.LockedBuffer();
    for(El::Int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(El::Int i = 0; i < A.Height(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(a[j * A.LDim() + i]);
        n[j] = sqrt(n[j]);
    }
}

template<typename T>
inline void ColumnNrm2(const El::DistMatrix<T, El::STAR, El::STAR>& A,
    El::DistMatrix<El::Base<T>, El::STAR, El::STAR>& N) {
    N.Resize(A.Width(), 1);
    T *n = N.Buffer();
    const T *a = A.LockedBuffer();
    for(El::Int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(El::Int i = 0; i < A.LocalHeight(); i++)
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
    const T *a = Al.LockedBuffer();
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
        const T *a = Al.LockedBuffer();
        for(El::Int j = 0; j < Al.Width(); j++)
            for(El::Int i = 0; i < Al.Height(); i++)
                n[A.GlobalCol(j)] +=
                    a[j * Al.LDim() + i] * El::Conj(a[j * Al.LDim() + i]);


        El::Zero(N);
        El::mpi::AllReduce(n.data(), N.Buffer(), A.Width(), MPI_SUM,
            A.DistComm());
        for(El::Int j = 0; j < A.Width(); j++)
            N.Set(j, 0, sqrt(N.Get(j, 0)));
    }

    El::mpi::Broadcast(N.Buffer(), A.Width(), A.Root(), A.CrossComm());
}


template<typename T>
inline void ColumnDot(const El::Matrix<T>& A, const El::Matrix<T>& B,
    El::Matrix<T>& N) {

    // TODO just assuming sizes are OK for now.

    T *n = N.Buffer();
    const T *a = A.LockedBuffer();
    const T *b = B.LockedBuffer();
    for(El::Int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(El::Int i = 0; i < A.Height(); i++)
            n[j] += a[j * A.LDim() + i] * El::Conj(b[j * B.LDim() + i]);
    }
}

template<typename T>
inline void ColumnDot(const El::DistMatrix<T, El::STAR, El::STAR>& A,
    const El::DistMatrix<T, El::STAR, El::STAR>& B,
    El::DistMatrix<T, El::STAR, El::STAR>& N) {

    // TODO just assuming sizes are OK for now.

    T *n = N.Buffer();
    const T *a = A.LockedBuffer();
    const T *b = B.LockedBuffer();
    for(El::Int j = 0; j < A.Width(); j++) {
        n[j] = 0.0;
        for(El::Int i = 0; i < A.LocalHeight(); i++)
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
    const T *a = Al.LockedBuffer();
    const El::Matrix<T> &Bl = B.LockedMatrix();
    const T *b = Bl.LockedBuffer();
    for(El::Int j = 0; j < Al.Width(); j++)
        for(El::Int i = 0; i < Al.Height(); i++)
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

    T *n = N.Buffer();
    const T *a = A.LockedBuffer();
    const T *b = B.LockedBuffer();
    for(El::Int i = 0; i < A.Height(); i++)
        n[i] = 0.0;

    for(El::Int j = 0; j < A.Width(); j++) {
        for(El::Int i = 0; i < A.Height(); i++)
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
        El::Gemm(El::ADJOINT, El::NORMAL, T(-2.0) * alpha, A, B, T(beta), C);

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

/**
 * C = beta * C + alpha * l1_distance_matrix(A, B)
 */
template<typename T>
void L1DistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::Matrix<T> &A, const El::Matrix<T> &B,
    T beta, El::Matrix<T> &C) {

    // TODO verify sizes

    const T *a = A.LockedBuffer();
    El::Int ldA = A.LDim();

    const T *b = B.LockedBuffer();
    El::Int ldB = B.LDim();

    T *c = C.Buffer();
    El::Int ldC = C.LDim();

    El::Int d = A.Height();

    /* Not the most efficient way... but mimicking BLAS is too much work! */
    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        for (El::Int j = 0; j < B.Width(); j++)
            for (El::Int i = 0; i < A.Width(); i++) {
                T v = 0.0;
                for (El::Int k = 0; k < d; k++)
                    v += std::abs(b[j * ldB + k] - a[i * ldA + k]);
                c[j * ldC + i] = beta * c[j * ldC + i] + alpha * v;
            }

    }

    // TODO the rest of the cases.
}

template<typename T>
void L1DistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::AbstractDistMatrix<T> &APre, const El::AbstractDistMatrix<T> &BPre,
    T beta, El::AbstractDistMatrix<T> &CPre) {

    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        // Use a SUMMA-like routine, with C as stationary
        // Basically an adaptation of Elementals TN case for stationary C.

        const El::Int m = CPre.Height();
        const El::Int n = CPre.Width();
        const El::Int sumDim = BPre.Height();
        const El::Int bsize = El::Blocksize();
        const El::Grid& g = APre.Grid();

        auto APtr = El::ReadProxy<T, El::MC, El::MR>(&APre);
        auto& A = *APtr;
        auto BPtr = El::ReadProxy<T, El::MC, El::MR>(&BPre);
        auto& B = *BPtr;
        auto CPtr = El::ReadWriteProxy<T, El::MC, El::MR>(&CPre);
        auto& C = *CPtr;

        // Temporary distributions
        El::DistMatrix<T, El::STAR, El::MC> A1_STAR_MC(g);
        El::DistMatrix<T, El::STAR, El::MR> B1_STAR_MR(g);

        A1_STAR_MC.AlignWith(C);
        B1_STAR_MR.AlignWith(C);

        El::Scale(beta, C);
        for(El::Int k = 0; k < sumDim; k += bsize) {
            const El::Int nb = std::min(bsize,sumDim-k);
            auto A1 = A(El::IR(k,k+nb), El::IR(0,m));
            auto B1 = B(El::IR(k,k+nb), El::IR(0,n));

            A1_STAR_MC = A1;
            B1_STAR_MR = B1;
            L1DistanceMatrix(base::COLUMNS, base::COLUMNS, alpha,
                A1_STAR_MC.LockedMatrix(), B1_STAR_MR.LockedMatrix(),
                T(1.0), C.Matrix());
        }
    }

    // TODO the rest of the cases.
}

/**
 * C = beta * C + alpha * l1_distance_matrix(A, A)
 * Update only lower part.
 */
template<typename T>
void SymmetricL1DistanceMatrix(El::UpperOrLower uplo, direction_t dir, T alpha,
    const El::Matrix<T> &A, T beta, El::Matrix<T> &C) {

    const T *a = A.LockedBuffer();
    El::Int ldA = A.LDim();

    T *c = C.Buffer();
    El::Int ldC = C.LDim();

    El::Int n = A.Width();
    El::Int d = A.Height();

    /* Not the most efficient way... but mimicking BLAS is too much work! */
    if (dir == base::COLUMNS) {
        for (El::Int j = 0; j < n; j++)
            for(El::Int i = ((uplo == El::UPPER) ? 0 : j);
                i < ((uplo == El::UPPER) ? j : n); i++)
            for (El::Int i = 0; i < A.Width(); i++) {
                T v = 0.0;
                for (El::Int k = 0; k < d; k++)
                    v += std::abs(a[j * ldA + k] - a[i * ldA + k]);
                c[j * ldC + i] = beta * c[j * ldC + i] + alpha * v;
            }

    }


    // TODO the rest of the cases.
}

template<typename T>
void SymmetricL1DistanceMatrix(El::UpperOrLower uplo, direction_t dir, T alpha,
    const El::AbstractDistMatrix<T> &A, T beta, El::AbstractDistMatrix<T> &C) {

    // TEMPORARY: do not take advantege of symmetry
    L1DistanceMatrix(dir, dir, alpha, A, A, beta, C);

}



} } // namespace skylark::base

#endif // SKYLARK_INNER_HPP
