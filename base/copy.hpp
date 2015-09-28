#ifndef SKYLARK_COPY_HPP
#define SKYLARK_COPY_HPP

namespace skylark { namespace base {

// TODO copy sparse to sparse.

template<typename T>
inline void Copy(const El::Matrix<T>& A, El::Matrix<T>& B) {
    El::Copy(A, B);
}

template<typename T>
inline void Copy(const El::AbstractDistMatrix<T>& A,
    El::AbstractDistMatrix<T>& B) {
    El::Copy(A, B);
}

/**
 * Copy matrix A into B, densifiying it in the process.
 */
template<typename T>
inline void DenseCopy(const sparse_matrix_t<T>& A, El::Matrix<T>& B) {
    El::Zeros(B, A.height(), A.width());

    const int *indptr = A.indptr();
    const int *indices = A.indices();
    const T *values = A.locked_values();
    for(int col = 0; col < A.width(); col++)
        for(int idx = indptr[col]; idx < indptr[col + 1]; idx++)
            B.Set(indices[idx], col, values[idx]);
}

/**
 * An alias to El::Copy
 */
template<typename T>
inline void DenseCopy(const El::Matrix<T>& A, El::Matrix<T>& B) {
    El::Copy(A, B);
}

template<typename T>
inline void DenseSubmatrixCopy(const El::Matrix<T>& A, El::Matrix<T> &B,
    El::Int i, El::Int j, El::Int height, El::Int width) {

    El::Matrix<T> Av = El::View(const_cast<El::Matrix<T>&>(A), i, j,
        height, width);
    El::Copy(Av, B);
}

template<typename T>
inline void DenseSubmatrixCopy(const sparse_matrix_t<T>& A, El::Matrix<T> &B,
    El::Int i, El::Int j, El::Int height, El::Int width) {

    El::Zeros(B, height, width);

    const int *indptr = A.indptr();
    const int *indices = A.indices();
    const T *values = A.locked_values();
    for(int col = j; col < j + width; col++)
        for(int idx = indptr[col]; idx < indptr[col + 1]; idx++)
            if (indices[idx] >= i && indices[idx] < i + height)
                B.Set(indices[idx] - i, col - j, values[idx]);
}


} } // namespace skylark::base

#endif // SKYLARK_COPY_HPP
