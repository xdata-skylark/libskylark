#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <boost/tuple/tuple.hpp>

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

    typedef ValueType value_type;

    typedef boost::tuple<int, int, value_type> coord_tuple_t;
    typedef std::vector<coord_tuple_t> coords_t;

    sparse_matrix_t()
        : _owndata(false), _dirty_struct(false), _height(0), _width(0), _nnz(0), 
          _indptr(nullptr), _indices(nullptr), _values(nullptr)
    {}

    ~sparse_matrix_t() {
        if (_owndata)
            _free_data();
    }

    bool struct_updated() const { return _dirty_struct; }
    void reset_dirty_struct() { _dirty_struct = false; }

    /**
     * Copy data to external buffers.
     */
    template<typename IdxType, typename ValType>
    void detach(IdxType *indptr, IdxType *indices, ValType *values) const {

        for(size_t i = 0; i < _width; ++i)
            indptr[i] = static_cast<IdxType>(_indptr[i]);

        for(size_t i = 0; i < _nnz; ++i) {
            indices[i] = static_cast<int32_t>(_indices[i]);
            values[i] = static_cast<ValType>(_values[i]);
        }
    }

    /**
     * Attach new structure and values.
     */
    void attach(int *indptr, int *indices, double *values,
        int nnz, int n_rows, int n_cols, bool _own = false) {
        if (_owndata)
            _free_data();

        _indptr = indptr;
        _indices = indices;
        _values = values;
        _nnz = nnz;
        _width = n_cols;
        _height = n_rows;

        _owndata = _own;
        _dirty_struct = true;
    }


    // attaching a coordinate structure facilitates going from distributed
    // input to local output.

    void set(coords_t coords, int n_rows = 0, int n_cols = 0) {


        sort(coords.begin(), coords.end(), &sparse_matrix_t::_sort_coords);

        _width = std::max(n_cols, boost::get<1>(coords.back()) + 1);
        _indptr = new int[_width + 1];

        // Count non-zeros
        _nnz = 0;
         for(size_t i = 0; i < coords.size(); ++i) {
             _nnz++;
             int cur_row = boost::get<0>(coords[i]);
             int cur_col = boost::get<1>(coords[i]);
             value_type cur_val = boost::get<2>(coords[i]);
             while(i + 1 < coords.size() &&
                 cur_row == boost::get<0>(coords[i + 1]) &&
                 cur_col == boost::get<1>(coords[i + 1]))
                 i++;
         }

         _indices = new int[_nnz];
         _values = new value_type[_nnz];

         _nnz = 0;
         int indptr_idx = 0;
         _indptr[indptr_idx] = 0;
         for(size_t i = 0; i < coords.size(); ++i) {
             _nnz++;
             _indptr[indptr_idx + 1]++;

            int cur_row = boost::get<0>(coords[i]);
            int cur_col = boost::get<1>(coords[i]);
            value_type cur_val = boost::get<2>(coords[i]);

            for(; indptr_idx < cur_col; ++indptr_idx)
                _indptr[indptr_idx + 1] = _nnz;

            // sum duplicates
            while(i + 1 < coords.size() &&
                  cur_row == boost::get<0>(coords[i + 1]) &&
                  cur_col == boost::get<1>(coords[i + 1])) {

                cur_val += boost::get<2>(coords[i + 1]);
                i++;
            }

            _indices[_nnz - 1] = cur_row;
            _values[_nnz - 1] = cur_val;

            n_rows = std::max(cur_row + 1, n_rows);
         }


         for(; indptr_idx < _width; ++indptr_idx)
             _indptr[indptr_idx + 1] = _nnz;

         _height = n_rows;
         _width = n_cols;

         _dirty_struct = true;
         _owndata = true;
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

    int Height() const {
        return height();
    }

    int Width() const {
        return width();
    }


    const int* indptr() const {
        return _indptr;
    }

    const int* indices() const {
        return _indices;
    }

    value_type* values() {
        return _values;
    }

    const value_type* locked_values() const {
        return _values;
    }

    bool operator==(const sparse_matrix_t &rhs) const {

        return (_indptr  == rhs._indptr) &&
               (_indices == rhs._indices) &&
               (_values  == rhs._values);
    }

private:
    bool _owndata;

    bool _dirty_struct;

    int _height;
    int _width;
    int _nnz;

    int* _indptr;
    int* _indices;
    value_type* _values;

    // TODO add the following
    sparse_matrix_t(const sparse_matrix_t&);
    void operator=(const sparse_matrix_t&);

    void _free_data() {
        delete[] _indptr;
        delete[] _indices;
        delete[] _values;
    }

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
