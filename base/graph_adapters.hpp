#ifndef SKYLARK_GRAPH_ADAPTERS_HPP
#define SKYLARK_GRAPH_ADAPTERS_HPP

namespace skylark { namespace base {

struct unweighted_local_graph_adapter_t {

    template<typename T>
    unweighted_local_graph_adapter_t(const sparse_matrix_t<T>& A)
        : _indptr(A.indptr()), _indices(A.indices()),
          _num_vertices(A.height()), _num_edges(A.nonzeros()) {

    }

    int num_vertices() const { return _num_vertices; }
    int num_edges() const { return _num_edges; }
    int degree(int vertex) const { return _indptr[vertex+1] - _indptr[vertex]; }
    const int *adjanct(int vertex) const { return _indices + _indptr[vertex]; }

private:
    const int *_indptr;
    const int *_indices;
    int _num_vertices;
    int _num_edges;
};

} } // namespace skylark::base

#endif // SKYLARK_GRAPH_ADAPTERS_HPP
