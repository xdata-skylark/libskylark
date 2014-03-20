#ifndef SKYLARK_VIEWING_HPP
#define SKYLARK_VIEWING_HPP

namespace skylark { namespace base {

template<typename T>
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

    A.attach(indptr, indices, values, indptr[width], A.height(), width, false);
}

template<typename T>
const sparse_matrix_t<T> ColumnView(const sparse_matrix_t<T>& B, int j, int width) {
    sparse_matrix_t<T> A;
    ColumnView(A, const_cast<sparse_matrix_t<T>&>(B), j, width);
    return A;
}

} } // namespace skylark::base

#endif // SKYLARK_VIEWING_HPP
