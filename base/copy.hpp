#ifndef SKYLARK_COPY_HPP
#define SKYLARK_COPY_HPP

namespace skylark { namespace base {

// TODO copy sparse to sparse.

template<typename T>
inline void Copy(const El::Matrix<T>& A, El::Matrix<T>& B) {
    El::Copy(A, B);
}

/**
 * Copy matrix A into B, densifiying it in the process.
 */
template<typename T>
inline void DenseCopy(const sparse_matrix_t<T>& A, El::Matrix<T>& B) {
    if (B.Height() != A.height() || B.Width() != A.width())
        B.Resize(A.height(), A.width());

    El::Zero(B);

    const int *indptr = A.indptr();
    const int *indices = A.indices();
    const double *values = A.locked_values();
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

} } // namespace skylark::base

#endif // SKYLARK_COPY_HPP
