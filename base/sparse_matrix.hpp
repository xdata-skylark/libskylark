#ifndef SKYLARK_SPARSE_MATRIX_HPP
#define SKYLARK_SPARSE_MATRIX_HPP

#include <set>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

namespace skylark { namespace base {

/**
 *  This implements a very crude CSC sparse matrix container only intended to
 *  hold local sparse matrices.
 *
 *  Row indices are not sorted.
 *  Structure is always constants, and can only be attached by Attached.
 *  Values of non-zeros can be modified.
 */
template<typename ValueType=double>
struct sparse_matrix_t {

    typedef int index_type;
    typedef ValueType value_type;

    typedef boost::tuple<index_type, index_type, value_type> coord_tuple_t;
    typedef std::vector<coord_tuple_t> coords_t;

    sparse_matrix_t()
        : _ownindptr(false), _ownindices(false), _ownvalues(false),
          _dirty_struct(false), _height(0), _width(0), _nnz(0),
          _indptr(nullptr), _indices(nullptr), _values(nullptr)
    {}

    // The following relies on C++11
    sparse_matrix_t(sparse_matrix_t<ValueType>&& A) :
        _ownindptr(A._ownindptr), _ownindices(A._ownindices),
        _ownvalues(A._ownvalues), _dirty_struct(A._dirty_struct),
        _height(A._height), _width(A._width), _nnz(A._nnz),
        _indptr(A._indptr), _indices(A._indices), _values(A._values)
    {
        A._ownindptr = false;
        A._ownindices = false;
        A._ownvalues = false;
    }

    ~sparse_matrix_t() {
        _free_data();
    }

    bool struct_updated() const { return _dirty_struct; }
    void reset_update_flag()    { _dirty_struct = false; }

    /**
     * Copy data to external buffers.
     */
    template<typename IdxType, typename ValType>
    void detach(IdxType *indptr, IdxType *indices, ValType *values) const {

        for(size_t i = 0; i <= _width; ++i)
            indptr[i] = static_cast<IdxType>(_indptr[i]);

        for(size_t i = 0; i < _nnz; ++i) {
            indices[i] = static_cast<IdxType>(_indices[i]);
            values[i] = static_cast<ValType>(_values[i]);
        }
    }

    /**
     * Attach new structure and values.
     */
    void attach(const index_type *indptr, const index_type *indices,
        value_type *values, int nnz, int n_rows, int n_cols, bool _own = false) {
        attach(indptr, indices, values, nnz, n_rows, n_cols, _own, _own, _own);
    }

    /**
     * Attach new structure and values.
     */
    void attach(const index_type *indptr, const index_type *indices,
        value_type *values, int nnz, int n_rows, int n_cols,
        bool ownindptr, bool ownindices, bool ownvalues) {
        _free_data();

        _indptr = indptr;
        _indices = indices;
        _values = values;
        _nnz = nnz;
        _width = n_cols;
        _height = n_rows;

        _ownindptr = ownindptr;
        _ownindices = ownindices;
        _ownvalues = ownvalues;

        _dirty_struct = true;
    }


    // attaching a coordinate structure facilitates going from distributed
    // input to local output.
    void set(coords_t coords, int n_rows = 0, int n_cols = 0) {

        sort(coords.begin(), coords.end(), &sparse_matrix_t::_sort_coords);

        n_cols = std::max(n_cols, boost::get<1>(coords.back()) + 1);
        index_type *indptr = new index_type[n_cols + 1];

        // Count non-zeros
        int nnz = 0;
        for(size_t i = 0; i < coords.size(); ++i) {
            nnz++;
            index_type cur_row = boost::get<0>(coords[i]);
            index_type cur_col = boost::get<1>(coords[i]);
            while(i + 1 < coords.size() &&
                  cur_row == boost::get<0>(coords[i + 1]) &&
                  cur_col == boost::get<1>(coords[i + 1]))
                i++;
        }

        index_type *indices = new index_type[nnz];
        value_type *values = new value_type[nnz];

        nnz = 0;
        int indptr_idx = 0;
        indptr[indptr_idx] = 0;
        for(size_t i = 0; i < coords.size(); ++i) {
            index_type cur_row = boost::get<0>(coords[i]);
            index_type cur_col = boost::get<1>(coords[i]);
            value_type cur_val = boost::get<2>(coords[i]);

            for(; indptr_idx < cur_col; ++indptr_idx)
                indptr[indptr_idx + 1] = nnz;
            nnz++;

            // sum duplicates
            while(i + 1 < coords.size() &&
                  cur_row == boost::get<0>(coords[i + 1]) &&
                  cur_col == boost::get<1>(coords[i + 1])) {

                cur_val += boost::get<2>(coords[i + 1]);
                i++;
            }

            indices[nnz - 1] = cur_row;
            values[nnz - 1] = cur_val;

            n_rows = std::max(cur_row + 1, n_rows);
        }

        for(; indptr_idx < n_cols; ++indptr_idx)
            indptr[indptr_idx + 1] = nnz;

        attach(indptr, indices, values, nnz, n_rows, n_cols, true);
    }

    int height() const {
        return _height;
    }

    int width() const {
        return _width;
    }

    int nonzeros() const {
        return _nnz;
    }

    const index_type* indptr() const {
        return _indptr;
    }

    const index_type* indices() const {
        return _indices;
    }

    value_type* values() {
        return _values;
    }

    const value_type* locked_values() const {
        return _values;
    }

    bool operator==(const sparse_matrix_t &rhs) const {

        // column pointer arrays have to be exactly the same
        if (std::vector<int>(_indptr, _indptr+_width) !=
            std::vector<int>(rhs._indptr, rhs._indptr + rhs._width))
            return false;

        // check more carefully for unordered row indices
        const int* indptr  = _indptr;
        const int* indices = _indices;
        const value_type* values = _values;

        const int* indices_rhs   = rhs.indices();
        const value_type* values_rhs = rhs.locked_values();

        for(int col = 0; col < width(); col++) {

            boost::unordered_map<int, value_type> col_values;

            for(int idx = indptr[col]; idx < indptr[col + 1]; idx++)
                col_values.insert(std::make_pair(indices[idx], values[idx]));

            for(int idx = indptr[col]; idx < indptr[col + 1]; idx++) {
                if(col_values[indices_rhs[idx]] != values_rhs[idx])
                    return false;
            }
        }

        return true;
    }

private:
    bool _ownindptr;
    bool _ownindices;
    bool _ownvalues;

    bool _dirty_struct;

    int _height;
    int _width;
    int _nnz;

    const index_type* _indptr;
    const index_type* _indices;
    value_type* _values;

    // TODO add the following
    sparse_matrix_t(const sparse_matrix_t&);
    void operator=(const sparse_matrix_t&);

    void _free_data() {
        if (_ownindptr)
            delete[] _indptr;
        if (_ownindices)
            delete[] _indices;
        if (_ownvalues)
            delete[] _values;
    }

    static bool _sort_coords(coord_tuple_t lhs, coord_tuple_t rhs) {
        if(boost::get<1>(lhs) != boost::get<1>(rhs))
            return boost::get<1>(lhs) < boost::get<1>(rhs);
        else
            return boost::get<0>(lhs) < boost::get<0>(rhs);
    }
};

template<typename T>
void Transpose(const sparse_matrix_t<T>& A, sparse_matrix_t<T>& B) {

    typedef typename sparse_matrix_t<T>::index_type index_type;
    typedef typename sparse_matrix_t<T>::value_type value_type;

    const index_type* aindptr = A.indptr();
    const index_type* aindices = A.indices();
    const value_type* avalues = A.locked_values();

    int m = A.width();
    int n = A.height();
    int nnz = A.nonzeros();

    index_type *indptr = new int[n + 1];
    index_type *indices = new int[nnz];
    value_type *values = new value_type[nnz];

    // Count nonzeros in each row
    index_type *nzrow = new int[n];
    std::fill(nzrow, nzrow + n, 0);
    for(int col = 0; col < m; col++)
        for(int idx = aindptr[col]; idx < aindptr[col + 1]; idx++)
            nzrow[aindices[idx]]++;

    // Set indptr
    indptr[0] = 0;
    for(int col = 1; col <= n; col++)
        indptr[col] = indptr[col - 1] + nzrow[col - 1];

    // Fill values
    std::fill(nzrow, nzrow + n, 0);
    for(int col = 0; col < m; col++)
        for(int idx = aindptr[col]; idx < aindptr[col + 1]; idx++) {
            int row = aindices[idx];
            T val = avalues[idx];
            indices[indptr[row] + nzrow[row]] = col;
            values[indptr[row] + nzrow[row]] = val;
            nzrow[row]++;
        }

    delete[] nzrow;

    B.attach(indptr, indices, values, nnz, m, n, true);
}

} }

#endif
