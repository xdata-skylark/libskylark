#ifndef SKYLARK_COPY_HPP
#define SKYLARK_COPY_HPP

namespace skylark { namespace base {

// TODO copy sparse to sparse.

#if SKYLARK_HAVE_ELEMENTAL

template<typename T>
inline void Copy(const elem::Matrix<T>& A, elem::Matrix<T>& B) {
    elem::Copy(A, B);
}

/**
 * Copy matrix A into B, densifiying it in the process.
 */
template<typename T>
inline void DenseCopy(const sparse_matrix_t<T>& A, elem::Matrix<T>& B) {
    if (B.Height() != A.height() || B.Width() != A.width())
        B.Resize(A.height(), A.width());

    elem::Zero(B);

    const int *indptr = A.indptr();
    const int *indices = A.indices();
    const double *values = A.locked_values();
    for(int col = 0; col < A.width(); col++)
        for(int idx = indptr[col]; idx < indptr[col + 1]; idx++)
            B.Set(indices[idx], col, values[idx]);
}

/**
 * An alias to elem::Copy
 */
template<typename T>
inline void DenseCopy(const elem::Matrix<T>& A, elem::Matrix<T>& B) {
    elem::Copy(A, B);
}

#endif

} } // namespace skylark::base

#endif // SKYLARK_COPY_HPP
