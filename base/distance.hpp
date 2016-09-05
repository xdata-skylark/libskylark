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

    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
            base::ml_exception()
            << base::error_msg(
            "EuclideanDistanceMatrix has not yet been implemented for that kind of matrices"));
    }


}

template<typename T>
void EuclideanDistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::ElementalMatrix<T> &A, const El::ElementalMatrix<T> &B,
    T beta, El::ElementalMatrix<T> &C) {

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
    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "EuclideanDistanceMatrix has not yet been implemented for that kind of matrices"));
    }


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
                i < ((uplo == El::UPPER) ? (j + 1) : n); i++)
                c[j * ldC + i] += alpha * (nn[i] * nn[i] + nn[j] * nn[j]);

    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "SymmetricEuclideanDistanceMatrix has not yet been implemented for that kind of matrices"));
    }

   
}

template<typename T>
void SymmetricEuclideanDistanceMatrix(El::UpperOrLower uplo, direction_t dir,
    T alpha, const El::ElementalMatrix<T> &A,
    T beta, El::ElementalMatrix<T> &C) {

    T *c = C.Buffer();
    int ldC = C.LDim();

    if (dir == base::COLUMNS) {
        El::Herk(uplo, El::ADJOINT, -2.0 * alpha, A, beta, C);
        //El::Gemm(El::ADJOINT, El::NORMAL, T(-2.0) * alpha, A, A, beta, C); 

        El::DistMatrix<El::Base<T>, El::STAR, El::STAR > N;
        ColumnNrm2(A, N);
        El::Base<T> *nn = N.Buffer();;

        El::Int n = C.LocalWidth();
        El::Int m = C.LocalHeight();

        for(El::Int j = 0; j < n; j++)
            for(El::Int i =
                    ((uplo == El::UPPER) ? 0 : C.LocalRowOffset(A.GlobalCol(j)));
                i < ((uplo == El::UPPER) ? C.LocalRowOffset(A.GlobalCol(j) + 1) : m); i++) {

                El::Base<T> a = nn[C.GlobalRow(i)];
                El::Base<T> b = nn[C.GlobalCol(j)];
                c[j * ldC + i] += alpha * (a * a + b * b);
            }
    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "SymmetricEuclideanDistanceMatrix has not yet been implemented for that kind of matrices"));
    }


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

    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "L1DistanceMatrix has not yet been implemented for that kind of matrices"));
    }


    
}

template<typename T>
void L1DistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::ElementalMatrix<T> &APre, const El::ElementalMatrix<T> &BPre,
    T beta, El::ElementalMatrix<T> &CPre) {

    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        // Use a SUMMA-like routine, with C as stationary
        // Basically an adaptation of Elementals TN case for stationary C.

        const El::Int m = CPre.Height();
        const El::Int n = CPre.Width();
        const El::Int sumDim = BPre.Height();
        const El::Int bsize = El::Blocksize();
        const El::Grid& g = APre.Grid();

        El::DistMatrixReadProxy<T, T, El::MC, El::MR> AProx(APre);
        El::DistMatrixReadProxy<T, T, El::MC, El::MR> BProx(BPre);
        El::DistMatrixReadWriteProxy<T, T, El::MC, El::MR> CProx(CPre);
        auto& A = AProx.GetLocked();
        auto& B = BProx.GetLocked();
        auto& C = CProx.Get();

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
    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "L1DistanceMatrix has not yet been implemented for that kind of matrices"));
    }
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
                i < ((uplo == El::UPPER) ? (j + 1) : n); i++)
            for (El::Int i = 0; i < A.Width(); i++) {
                T v = 0.0;
                for (El::Int k = 0; k < d; k++)
                    v += std::abs(a[j * ldA + k] - a[i * ldA + k]);
                c[j * ldC + i] = beta * c[j * ldC + i] + alpha * v;
            }

    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "SymmetricL1DistanceMatrix has not yet been implemented for that kind of matrices"));
    }

}

namespace internal {

/**
 * Same as L1DistanceMatrix, except only a traingular part is updated
 * (hence the TU)
 */
template<typename T>
void L1DistanceMatrixTU(El::UpperOrLower uplo,
    direction_t dirA, direction_t dirB, T alpha,
    const El::DistMatrix<T, El::STAR, El::MC> &A,
    const El::DistMatrix<T, El::STAR, El::MR> &B,
    T beta, El::DistMatrix<T> &C) {

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
        El::Int n = C.LocalWidth();
        El::Int m = C.LocalHeight();
        for (El::Int j = 0; j < n; j++)
            for(El::Int i =
                    ((uplo == El::UPPER) ? 0 : C.LocalRowOffset(A.GlobalCol(j)));
                i < ((uplo == El::UPPER) ? C.LocalRowOffset(A.GlobalCol(j) + 1) : m); i++) {

                T v = 0.0;
                for (El::Int k = 0; k < d; k++)
                    v += std::abs(b[j * ldB + k] - a[i * ldA + k]);
                c[j * ldC + i] = beta * c[j * ldC + i] + alpha * v;
            }

    } else {
            // TODO the rest of the cases.
    SKYLARK_THROW_EXCEPTION (
    base::ml_exception()
        << base::error_msg(
        "L1DistanceMatrixTU has not yet been implemented for that kind of matrices"));
    }
}} // namespace internal

template<typename T>
void SymmetricL1DistanceMatrix(El::UpperOrLower uplo, direction_t dir, T alpha,
    const El::ElementalMatrix<T> &APre, T beta, El::ElementalMatrix<T> &CPre) {

    if (dir == base::COLUMNS) {

        const El::Int r = APre.Height();
        const El::Int bsize = El::Blocksize();
        const El::Grid& g = APre.Grid();

        El::DistMatrixReadProxy<T, T, El::MC, El::MR> AProx(APre);
        El::DistMatrixReadWriteProxy<T, T, El::MC, El::MR> CProx(CPre);
        auto& A = AProx.GetLocked();
        auto& C = CProx.Get();

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
                alpha, A1_STAR_MC, A1_STAR_MR,
                T(1.0), C);
        }
    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "SymmetricL1DistanceMatrix has not yet been implemented for that kind of matrices"));
    }
}


/**
 * C = beta * C + alpha * expsemigroupDistanceMatrix(A, B)
 */
template<typename T>
void ExpsemigroupDistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::Matrix<T> &A, const El::Matrix<T> &B,
    T beta, El::Matrix<T> &C) {

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
                    v += std::sqrt(std::abs(b[j * ldB + k] + a[i * ldA + k]));
                c[j * ldC + i] = beta * c[j * ldC + i] + alpha * v;
            }

    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "ExpsemigroupDistanceMatrix has not yet been implemented for that kind of matrices"));
    }
}

template<typename T>
void ExpsemigroupDistanceMatrix(direction_t dirA, direction_t dirB, T alpha,
    const El::ElementalMatrix<T> &APre, const El::ElementalMatrix<T> &BPre,
    T beta, El::ElementalMatrix<T> &CPre) {

    if (dirA == base::COLUMNS && dirB == base::COLUMNS) {
        // Use a SUMMA-like routine, with C as stationary
        // Basically an adaptation of Elementals TN case for stationary C.

        const El::Int m = CPre.Height();
        const El::Int n = CPre.Width();
        const El::Int sumDim = BPre.Height();
        const El::Int bsize = El::Blocksize();
        const El::Grid& g = APre.Grid();

        El::DistMatrixReadProxy<T, T, El::MC, El::MR> AProx(APre);
        El::DistMatrixReadProxy<T, T, El::MC, El::MR> BProx(BPre);
        El::DistMatrixReadWriteProxy<T, T, El::MC, El::MR> CProx(CPre);
        auto& A = AProx.GetLocked();
        auto& B = BProx.GetLocked();
        auto& C = CProx.Get();

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
            ExpsemigroupDistanceMatrix(base::COLUMNS, base::COLUMNS, alpha,
                A1_STAR_MC.LockedMatrix(), B1_STAR_MR.LockedMatrix(),
                T(1.0), C.Matrix());
        }
    } else {
        // TODO the rest of the cases.
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
            << base::error_msg(
            "ExpsemigroupDistanceMatrix has not yet been implemented for that kind of matrices"));
    }


}

/**
 * C = beta * C + alpha * expsemigroupDistanceMatrix(A, B)
 * Update only lower part.
 */
template<typename T>
void SymmetricExpsemigroupDistanceMatrix(El::UpperOrLower uplo, direction_t dir,
    T alpha, const El::Matrix<T> &A, T beta, El::Matrix<T> &C) {

    // TODO the rest of the cases.
    SKYLARK_THROW_EXCEPTION (
    base::ml_exception()
        << base::error_msg(
        "SymmetricExpsemigroupDistanceMatrix has not yet been implemented for that kind of matrices"));
}

template<typename T>
void SymmetricExpsemigroupDistanceMatrix(El::UpperOrLower uplo, direction_t dir,
    T alpha, const El::ElementalMatrix<T> &A,
    T beta, El::ElementalMatrix<T> &C) {


    // TODO the rest of the cases.
    SKYLARK_THROW_EXCEPTION (
    base::ml_exception()
        << base::error_msg(
        "SymmetricExpsemigroupDistanceMatrix has not yet been implemented for that kind of matrices"));
}

} } // namespace skylark::base

#endif // SKYLARK_DISTANCE_HPP
