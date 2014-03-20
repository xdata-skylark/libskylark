#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <boost/tuple/tuple.hpp>

namespace skylark { namespace base {

/**
 *  This implements a very crude CSC sparse matrix container only intended to
 *  hold local sparse matrices.
 */
template<typename ValueType=double>
struct sparse_matrix_t {

    typedef ValueType value_type;

    typedef typename std::vector<int>::const_iterator const_ind_itr_t;
    typedef std::pair<const_ind_itr_t, const_ind_itr_t> const_ind_itr_range_t;

    typedef typename std::vector<ValueType>::const_iterator const_val_itr_t;
    typedef std::pair<const_val_itr_t, const_val_itr_t> const_val_itr_range_t;

    typedef boost::tuple<int, int, value_type> coord_tuple_t;
    typedef std::vector<coord_tuple_t> coords_t;

    sparse_matrix_t()
        : _dirty(false)
    {_height=0; _width=0;}

    ~sparse_matrix_t() {}

    // expose size to allocate arrays for detach
    void get_size(int *n_indptr, int *n_indices) const {

        *n_indptr  = _indptr.size();
        *n_indices = _indices.size();
    }

    // expose dirty flag to CAPI
    bool needs_update() const { return _dirty; }

    // expose to the CAPI layer assume memory has been allocated
    void detach(int32_t *indptr, int32_t *indices, double *values) const {

        for(size_t i = 0; i < _indptr.size(); ++i)
            indptr[i] = static_cast<int32_t>(_indptr[i]);

        for(size_t i = 0; i < _indices.size(); ++i)
            indices[i] = static_cast<int32_t>(_indices[i]);

        for(size_t i = 0; i < _values.size(); ++i)
            values[i] = static_cast<double>(_values[i]);
    }

    // attach data from CAPI
    void attach(int *indptr, int *indices, double *values,
        int n_indptr, int n_ind, int n_rows, int n_cols) {

        //XXX: we could use pointer for indptr array, indicies and values
        //     array only determined later by the nnz.
        _indptr.assign(indptr, indptr + n_indptr);
        _indices.assign(indices, indices + n_ind);
        _values.assign(values, values + n_ind);

        _width = n_cols;
        _height = n_rows;

        //XXX: assume this is only called from the python layer
        _dirty = false;
    }


    // attaching a coordinate structure facilitates going from distributed
    // input to local output.
    void attach(coords_t coords, int n_rows = 0, int n_cols = 0) {

        _indptr.clear();
        _indices.clear();
        _values.clear();

        sort(coords.begin(), coords.end(), &sparse_matrix_t::_sort_coords);

        _indptr.push_back(0);
        int indptr_idx = 0;
        for(size_t i = 0; i < coords.size(); ++i) {
            int cur_row = boost::get<0>(coords[i]);
            int cur_col = boost::get<1>(coords[i]);
            value_type cur_val = boost::get<2>(coords[i]);

            for(; indptr_idx < cur_col; ++indptr_idx)
                _indptr.push_back(_indices.size());

            // sum duplicates
            while(i + 1 < coords.size() &&
                  cur_row == boost::get<0>(coords[i + 1]) &&
                  cur_col == boost::get<1>(coords[i + 1])) {

                cur_val += boost::get<2>(coords[i + 1]);
                i++;
            }

            _indices.push_back(cur_row);
            _values.push_back(cur_val);

            n_rows = std::max(cur_row + 1, n_rows);
        }

        // in case we specified the cols fill possible empty rows in the end.
        if(n_cols > 0)
            for(; indptr_idx < n_cols + 1; ++indptr_idx)
                _indptr.push_back(_indices.size());
        else
            _indptr.push_back(_indices.size());

        _height = n_rows;
        _width = n_cols;

        _dirty = !_dirty;
    }

    const_ind_itr_range_t indptr_itr() const {
        return std::make_pair(_indptr.begin(), _indptr.end());
    }

    int height() const {
        return _height;
    }

    int width() const {
        return _width;
    }

    int Height() const {
        return _height;
    }

    int Width() const {
        return _width;
    }


    const_ind_itr_range_t indices_itr() const {
        return std::make_pair(_indices.begin(), _indices.end());
    }

    const_val_itr_range_t values_itr() const {
        return std::make_pair(_values.begin(), _values.end());
    }

    std::vector<int>& indptr() {
        return _indptr;
    }

    std::vector<int>& indices() {
        return _indices;
    }

    std::vector<value_type>& values() {
        return _values;
    }

    const std::vector<int>& locked_indptr() const {
        return _indptr;
    }

    const std::vector<int>& locked_indices() const {
        return _indices;
    }

    const std::vector<value_type>& locked_values() const {
        return _values;
    }

    bool operator==(const sparse_matrix_t &rhs) const {

        return (_indptr  == rhs._indptr) &&
               (_indices == rhs._indices) &&
               (_values  == rhs._values);
    }

private:
    bool _dirty;

    int _height;
    int _width;

    std::vector<int> _indptr;
    std::vector<int> _indices;
    std::vector<value_type> _values;

    sparse_matrix_t(const sparse_matrix_t&);
    void operator=(const sparse_matrix_t&);

    static bool _sort_coords(coord_tuple_t lhs, coord_tuple_t rhs) {
        if(boost::get<1>(lhs) != boost::get<1>(rhs))
            return boost::get<1>(lhs) < boost::get<1>(rhs);
        else
            return boost::get<0>(lhs) < boost::get<0>(rhs);
    }
};

}
}

#endif
