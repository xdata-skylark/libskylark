#ifndef SKYLARK_DISTANCE_HPP
#define SKYLARK_DISTANCE_HPP


namespace skylark { namespace base {

/**
 * C = beta * C + alpha * square_euclidean_distance_matrix(A, B)
 */
template<typename T>
void EuclideanDistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::Matrix<T> &A, const El::Matrix<T> &B,
    T beta, El::Matrix<T> &C) {

    T *c = C.Buffer();
    El::Int ldC = C.LDim();

    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        base::Gemm(El::ADJOINT, El::NORMAL, T(-2.0) * alpha, A, B, beta, C);

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
void EuclideanDistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
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
void SymmetricEuclideanDistanceMatrix(El::UpperOrLower uplo, direction_t dir,
    T alpha, const El::Matrix<T> &A, T beta, El::Matrix<T> &C) {

    T *c = C.Buffer();
    int ldC = C.LDim();

    if (dir == base::COLUMNS) {
        El::Herk(uplo, El::ADJOINT, -2.0 * alpha, A, beta, C);
        //El::Gemm(El::ADJOINT, El::NORMAL, T(-2.0) * alpha, A, A, beta, C);

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
void SymmetricEuclideanDistanceMatrix(El::UpperOrLower uplo, direction_t dir,
    T alpha, const El::AbstractDistMatrix<T> &A,
    T beta, El::AbstractDistMatrix<T> &C) {

    T *c = C.Buffer();
    int ldC = C.LDim();

    if (dir == base::COLUMNS) {
        El::Herk(uplo, El::ADJOINT, -2.0 * alpha, A, beta, C);
        //El::Gemm(El::ADJOINT, El::NORMAL, T(-2.0) * alpha, A, A, beta, C); 

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

namespace internal {

/**
 * Same as L1DistanceMatrix, except only a traingular part is updated
 * (hence the TU)
 */
template<typename T>
void L1DistanceMatrixTU(El::UpperOrLower uplo, 
    direction_t dirA, direction_t dirB, T alpha,
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
        El::Int n = A.Width();
        for (El::Int j = 0; j < n; j++)
            for(El::Int i = ((uplo == El::UPPER) ? 0 : j);
                i < ((uplo == El::UPPER) ? j : n); i++) {
                T v = 0.0;
                for (El::Int k = 0; k < d; k++)
                    v += std::abs(b[j * ldB + k] - a[i * ldA + k]);
                c[j * ldC + i] = beta * c[j * ldC + i] + alpha * v;
            }

    }

    // TODO the rest of the cases.
}

}

template<typename T>
void SymmetricL1DistanceMatrix(El::UpperOrLower uplo, direction_t dir, T alpha,
    const El::AbstractDistMatrix<T> &APre, T beta, El::AbstractDistMatrix<T> &CPre) {

    if (dir == base::COLUMNS) {

        const El::Int r = APre.Height();
        const El::Int bsize = El::Blocksize();
        const El::Grid& g = APre.Grid();

        auto APtr = El::ReadProxy<T, El::MC, El::MR>(&APre);
        auto& A = *APtr;
        auto CPtr = El::ReadWriteProxy<T, El::MC, El::MR>(&CPre);
        auto& C = *CPtr;

        // Temporary distributions
        El::DistMatrix<T, El::STAR, El::MR> A1_STAR_MR(g);
        El::DistMatrix<T, El::STAR, El::MC> A1_STAR_MC(g);

        A1_STAR_MC.AlignWith(C);
        A1_STAR_MR.AlignWith(C);

        El::ScaleTrapezoid(beta, uplo, C);
        for(El::Int k = 0; k < r; k += bsize) {
            const El::Int nb = std::min(bsize, r - k);
            auto A1 = A(El::IR(k, k + nb), El::ALL);

            A1_STAR_MC = A1;
            A1_STAR_MR = A1;

            internal::L1DistanceMatrixTU(uplo, base::COLUMNS, base::COLUMNS, 
                alpha, A1_STAR_MC.LockedMatrix(), A1_STAR_MR.LockedMatrix(),
                T(1.0), C.Matrix());
        }
    }

    // TODO the rest of the cases.
}



} } // namespace skylark::base

#endif // SKYLARK_DISTANCE_HPP
