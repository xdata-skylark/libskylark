#ifndef SKYLARK_VIEWING_HPP
#define SKYLARK_VIEWING_HPP

namespace skylark { namespace base {

#if SKYLARK_HAVE_ELEMENTAL

template<typename T>
inline void ColumnView(elem::Matrix<T>& A, elem::Matrix<T>& B,
    int j, int width) {
    elem::View(A, B, 0, j, B.Height(), width);
}

template<typename T>
inline
const elem::Matrix<T> ColumnView(const elem::Matrix<T>& B, int j, int width) {
    elem::Matrix<T> A;
    elem::LockedView(A, B, 0, j, B.Height(), width);
    return A;
}

template<typename T>
inline void RowView(elem::Matrix<T>& A, elem::Matrix<T>& B,
    int i, int height) {
    elem::View(A, B, i, 0, height, B.Width());
}

template<typename T>
inline
const elem::Matrix<T> RowView(const elem::Matrix<T>& B, int i, int height) {
    elem::Matrix<T> A;
    elem::LockedView(A, B, i, 0, height, B.Width());
    return A;
}
#endif

template<typename T>
inline
void ColumnView(sparse_matrix_t<T>& A, sparse_matrix_t<T>& B, int j, int width) {
    const int *bindptr = B.indptr();
    const int *bindices = B.indices();
    double *bvalues = B.values();

    int start = bindptr[j];
    int *indptr = new int[width + 1];
    for (int i = 0; i <= width; i++)
        indptr[i] = bindptr[j + i] - start;
    const int *indices = bindices + start;
    double *values = bvalues + start;

    A.attach(indptr, indices, values, indptr[width], A.height(), width,
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
