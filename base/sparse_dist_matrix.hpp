#ifndef SKYLARK_SPARSE_DIST_MATRIX_HPP
#define SKYLARK_SPARSE_DIST_MATRIX_HPP

#include <map>
#include <memory>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

#include <El.hpp>

#include "sparse_matrix.hpp"

namespace skylark { namespace base {

namespace detail {

    struct compare_t {
        bool operator()(const std::pair<int, int>& x, const std::pair<int, int>& y) {
            if(x.second == y.second)
                return x.first < y.first;

            return x.second < y.second;
        }
    };
}


/**
 *  This implements a very crude CSC sparse matrix container only intended to
 *  hold local sparse matrices.
 *
 *  Row indices are not sorted.
 *  Structure is always constants, and can only be attached by Attached.
 *  Values of non-zeros can be modified.
 */
template<typename ValueType=double>
struct sparse_dist_matrix_t {

    typedef int index_type;
    typedef ValueType value_type;

    //FIXME: comm?
    sparse_dist_matrix_t()
        : _local_buffer(new sparse_matrix_t<value_type>())
        , _comm(boost::mpi::communicator())
        , _grid(El::Grid(_comm))
        , _finalized(false)
        , _rank(_comm.rank())
        , _num_procs(_comm.size())
        , _n_rows(0)
        , _n_cols(0)
        , _n_local_rows(0)
        , _n_local_cols(0)
    {}

    sparse_dist_matrix_t(
            El::Int n_rows, El::Int n_cols, boost::mpi::communicator& comm,
            const El::Grid& grid)
        : _local_buffer(new sparse_matrix_t<value_type>())
        , _comm(comm)
        , _grid(grid)
        , _finalized(false)
        , _rank(comm.rank())
        , _num_procs(comm.size())
        , _n_rows(n_rows)
        , _n_cols(n_cols)
        , _n_local_rows(0)
        , _n_local_cols(0)
    {}

    ~sparse_dist_matrix_t()
    {}

    boost::mpi::communicator comm() const {
        return _comm;
    }

    void resize(El::Int n_rows, El::Int n_cols) {

        //FIXME: later ignore values that are outside
        assert(_finalized == false);
        _n_rows = n_rows;
        _n_cols = n_cols;
    }

    void queue_update(El::Int i, El::Int j, value_type value) {

        assert(_finalized == false);
        assert(i < height());
        assert(j < width());

        if(is_local(i, j)) {
            queue_update_local(local_row(i), local_col(j), value);
        }
    }

    void queue_update_local(El::Int i, El::Int j, value_type value) {

        assert(_finalized == false);

        _n_local_rows = std::max(_n_local_rows, i);
        _n_local_cols = std::max(_n_local_cols, j);

        //XXX: we should use a nicer structure for the temporary storage..
        _temp_buffer[std::make_pair(i, j)] += value;
    }

    //FIXME: use a more adequate structure
    void finalize() {

        assert(_finalized == false);

        _n_local_rows++;
        _n_local_cols++;
        _nnz = _temp_buffer.size();

        _indptr.resize(_n_local_cols + 1);
        _indices.resize(_nnz);
        _values.resize(_nnz);

        int nnz = 0;
        int indptr_idx = 0;
        _indptr[indptr_idx] = 0;

        typename std::map< std::pair<int, int>, value_type >::const_iterator itr;
        for(itr = _temp_buffer.begin(); itr != _temp_buffer.end(); itr++) {

            int cur_row = itr->first.first;
            int cur_col = itr->first.second;
            value_type cur_val = itr->second;

            // fill empty cols
            for(; indptr_idx < cur_col; ++indptr_idx)
                _indptr[indptr_idx + 1] = nnz;

            _indices[nnz] = cur_row;
            _values[nnz]  = cur_val;
            nnz++;
        }

        for(; indptr_idx < _n_local_cols; ++indptr_idx)
            _indptr[indptr_idx + 1] = nnz;

        assert(nnz == _nnz);
        _temp_buffer.clear();

        _local_buffer->attach(&_indptr[0], &_indices[0], &_values[0],
                _nnz, _n_local_rows, _n_local_cols, false);

        _global_nnz = 0;
        boost::mpi::all_reduce(_comm, _nnz, _global_nnz, std::plus<int>());

        _finalized = true;
    }


    El::Int height()   const { return _n_rows; }
    El::Int width()    const { return _n_cols; }
    El::Int nonzeros() const { return _global_nnz; }

    El::Int global_row(El::Int iLoc) const {
        return _col_shift + iLoc * _col_stride;
    }

    El::Int global_col(El::Int jLoc) const {
        return _row_shift + jLoc * _row_stride;
    }


    El::Int local_height()   const { return _local_buffer->height(); }
    El::Int local_width()    const { return _local_buffer->width(); }
    El::Int local_nonzeros() const { return _local_buffer->nonzeros(); }

    El::Int local_row_offset(El::Int i) const {
        return El::Length_(i, _col_shift, _col_stride);
    }

    El::Int local_col_offset(El::Int j) const {
        return El::Length_(j, _row_shift, _row_stride);
    }

    El::Int local_row(El::Int i) const {
        return local_row_offset(i);
    }

    El::Int local_col(El::Int j) const {
        return local_col_offset(j);
    }

    bool is_local_row(El::Int i) const { return row_owner(i) == _col_rank; }
    bool is_local_col(El::Int j) const { return col_owner(j) == _row_rank; }

    bool is_local(El::Int i, El::Int j) const {
        return is_local_row(i) && is_local_col(j);
    }

    bool is_finalized() const { return _finalized; }



    int row_owner(El::Int i) const {
        return int((i + _col_align) % _col_stride);
    }

    int col_owner(El::Int j) const {
        return int((j + _row_align) % _row_stride);
    }

    int owner(El::Int i, El::Int j) const {
        return row_owner(i) + col_owner(j) * _col_stride;
    }

    const index_type* indptr() const {
        if(!_finalized) return NULL;
        return _local_buffer->indptr();
    }

    const index_type* indices() const {
        if(!_finalized) return NULL;
        return _local_buffer->indices();
    }

    value_type* values() {
        if(!_finalized) return NULL;
        return _local_buffer->values();
    }

    const value_type* locked_values() const {
        if(!_finalized) return NULL;
        return _local_buffer->values();
    }



private:

    std::unique_ptr< sparse_matrix_t<value_type> > _local_buffer;
    std::map< std::pair<int, int>, value_type, detail::compare_t > _temp_buffer;

    const boost::mpi::communicator& _comm;
    const El::Grid& _grid;

    bool _finalized;
    std::vector<int> _indptr;
    std::vector<int> _indices;
    std::vector<value_type> _values;

protected:

    int _rank;
    int _num_procs;

    El::Int _n_rows;
    El::Int _n_cols;

    El::Int _n_local_rows;
    El::Int _n_local_cols;

    int _nnz;
    int _global_nnz;

    int _col_rank;
    int _row_rank;

    int _row_align;
    int _row_shift;
    int _row_stride;
    int _col_align;
    int _col_shift;
    int _col_stride;
};

} }

#endif
