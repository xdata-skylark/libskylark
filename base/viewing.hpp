#ifndef SKYLARK_VIEWING_HPP
#define SKYLARK_VIEWING_HPP

namespace skylark { namespace base {

template<typename T>
inline void ColumnView(El::Matrix<T>& A, El::Matrix<T>& B,
    int j, int width) {
    El::View(A, B, 0, j, B.Height(), width);
}

template<typename T>
inline
const El::Matrix<T> ColumnView(const El::Matrix<T>& B, int j, int width) {
    El::Matrix<T> A;
    El::LockedView(A, B, 0, j, B.Height(), width);
    return A;
}

template<typename T, El::Distribution U, El::Distribution V>
inline
const El::DistMatrix<T,U,V> ColumnView(const El::DistMatrix<T,U,V>& B,
    int j, int width) {
    El::DistMatrix<T,U,V> A;
    El::LockedView(A, B, 0, j, B.Height(), width);
    return A;
}

template<typename T>
inline void RowView(El::Matrix<T>& A, El::Matrix<T>& B,
    int i, int height) {
    El::View(A, B, i, 0, height, B.Width());
}

template<typename T>
inline
const El::Matrix<T> RowView(const El::Matrix<T>& B, int i, int height) {
    El::Matrix<T> A;
    El::LockedView(A, B, i, 0, height, B.Width());
    return A;
}

template<typename T>
inline
void ColumnView(sparse_matrix_t<T>& A, sparse_matrix_t<T>& B, int j, int width) {
    const int *bindptr = B.indptr();
    const int *bindices = B.indices();
    T *bvalues = B.values();

    int start = bindptr[j];
    int *indptr = new int[width + 1];
    for (int i = 0; i <= width; i++)
        indptr[i] = bindptr[j + i] - start;
    const int *indices = bindices + start;
    T *values = bvalues + start;

    A.attach(indptr, indices, values, indptr[width], B.height(), width,
        true, false, false);
}

template<typename T>
inline
sparse_matrix_t<T> ColumnView(const sparse_matrix_t<T>& B, int j, int width) {
    sparse_matrix_t<T> A;
    ColumnView(A, const_cast<sparse_matrix_t<T>&>(B), j, width);
    return A;
}

} } // namespace skylark::base

#endif // SKYLARK_VIEWING_HPP
