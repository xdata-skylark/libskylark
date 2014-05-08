#ifndef SKYLARK_COMBBLAS_SLAB_VIEW_HPP
#define SKYLARK_COMBBLAS_SLAB_VIEW_HPP

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#include <CommGrid.h>
#endif

#include <map>
#include <boost/mpi.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include "combblas_comm_grid.hpp"

namespace skylark {
namespace utility {

#if SKYLARK_HAVE_COMBBLAS

/**
 * Slab view for CombBLAS matrices.
 * This are helpers for the panel gemms, where we communicate parts of
 * the CombBLAS matrix and then perform a local Elemental gemm.
 * The communication patterns are dictated by the "output" Elemental
 * distribution, e.g. if we want an STAR / MR view, CombBLAS values are
 * redistributed accordingly.
 * Note that due to the CombBLAS distribution we generate slabs in parallel:
 *
 *     [x....|x....]
 *     [x....|x....]
 *     [-----------]
 *     [x....|x....]
 *     [x....|x....]
 *
 * FIXME: - see how we can integrate into base::view
 *        - different comm patterns for _set_view
 *        - compare performance against one-sided value getter for cb matrices
 *        - flag to generate in-order
 *        - transpose case should invert row/col slab view
 */
template <typename index_type, typename value_type>
struct combblas_slab_view_t {

    typedef SpDCCols<index_type, value_type> cb_col_t;
    typedef SpParMat<index_type, value_type, cb_col_t> sp_par_mat_t;
    typedef typename cb_col_t::SpColIter cb_col_itr_t;
    typedef typename cb_col_t::SpColIter::NzIter cb_nz_itr_t;

    /**
     *  Create a slab view of the CombBLAS matrix B that may be transposed.
     */
    combblas_slab_view_t(const sp_par_mat_t &B, bool transp_b = false)
        : _world(B.getcommgrid()->GetWorld(), boost::mpi::comm_duplicate)
        , _data((const_cast<sp_par_mat_t &>(B)).seq())
        , _col_itr(_data.begcol())
        , _row_itr(_data.begnz(_col_itr))
        , _transp_b(transp_b)
        , _offset_row(cb_my_row_offset(B))
        , _offset_col(cb_my_col_offset(B))
        , _global_row(0)
        , _global_col(0) {

        _ncol = B.getncol();
        _nrow = B.getnrow();
        if(_transp_b) {
            _ncol = B.getnrow();
            _nrow = B.getncol();
        }

        for(cb_col_itr_t col = _data.begcol(); col != _data.endcol(); col++)
            _row_itrs.push_back(_data.begnz(col));
    }

    /**
     * Extract the value of a global index from the CombBLAS matrix.
     * TODO: Use this to define a sparse-dense local GEMM that avoids building
     *       a dense part of the CombBLAS matrix.
     */
    value_type operator()(index_type g_row, index_type g_col) {

        index_type map_idx = idx(g_row, g_col);
        if(_values.count(map_idx) > 0)
            return _values[map_idx];
        else
            return static_cast<value_type>(0);
    }

#ifdef SKYLARK_HAVE_ELEMENTAL
    /**
     *  Extract a Elemental view of CombBLAS COLUMNS. The distribution of the
     *  Elemental view is determined by the distribution of the input matrix
     *  A.
     *  The column_idxs store the global column indices that contain non-zeros.
     */
    template<typename dist_elem_matrix_t>
    void extract_elemental_column_slab_view(
            const dist_elem_matrix_t &A,
            std::set<index_type> &column_idxs,
            size_t width = 1) {

        _values.clear();

        // structure per proc data
        std::vector< std::map<index_type, value_type> >
            per_proc_data(_world.size());

        // accumulate locale values
        for(size_t i = 0; i < width && _col_itr != _data.endcol();
            _col_itr++, i++) {

            // first accumulate column values, then distribute
            for(cb_nz_itr_t nz = _data.begnz(_col_itr);
                nz != _data.endnz(_col_itr); nz++) {

                index_type g_cb_col_idx = _col_itr.colid() + _offset_col;
                index_type g_cb_row_idx = nz.rowid()  + _offset_row;
                if(_transp_b) std::swap(g_cb_row_idx, g_cb_col_idx);

                size_t target_proc = A.Owner(g_cb_row_idx, g_cb_col_idx);
                index_type coords = idx(g_cb_row_idx, g_cb_col_idx);
                per_proc_data[target_proc].insert(
                    std::make_pair(coords, nz.value()));
            }
        }

        std::set<index_type> indices;
        _set_view(per_proc_data, indices);
        typename std::set<index_type>::iterator itr;
        for(itr = indices.begin(); itr != indices.end(); itr++)
            column_idxs.insert(col(*itr));
    }

    /**
     *  Extract a Elemental view of CombBLAS COLUMNS in sequence. The
     *  distribution of the Elemental view is determined by the distribution
     *  of the input matrix A.
     */
    template<typename dist_elem_matrix_t>
    void extract_elemental_column_slab_view(
            const dist_elem_matrix_t &A,
            size_t width = 1) {

        _values.clear();

        // structure per proc data
        std::vector< std::map<index_type, value_type> >
            per_proc_data(_world.size());

        for(size_t i = 0; i < width && _global_col < _ncol;
            _global_col++, i++) {

            // if I don't own anything in this column, continue
            if(_col_itr.colid() + _offset_col != _global_col)
                    continue;

            // first accumulate column values, then distribute
            for(cb_nz_itr_t nz = _data.begnz(_col_itr);
                nz != _data.endnz(_col_itr); nz++) {

                index_type g_cb_col_idx = _col_itr.colid() + _offset_col;
                index_type g_cb_row_idx = nz.rowid()  + _offset_row;
                if(_transp_b) std::swap(g_cb_row_idx, g_cb_col_idx);

                size_t target_proc = A.Owner(g_cb_row_idx, g_cb_col_idx);
                index_type coords = idx(g_cb_row_idx, g_cb_col_idx);
                per_proc_data[target_proc].insert(
                    std::make_pair(coords, nz.value()));
            }

            // this proc can advance to next column
            _col_itr++;
        }

        std::set<index_type> indices;
        _set_view(per_proc_data, indices);
    }

    /**
     *  Extract a Elemental view of CombBLAS ROWS. The distribution of the
     *  Elemental view is determined by the distribution of the input matrix
     *  A.
     *  The row_idxs store the global row indices that contain non-zeros.
     */
    template<typename dist_elem_matrix_t>
    void extract_elemental_row_slab_view(
            const dist_elem_matrix_t &A,
            std::set<index_type> &row_idxs,
            size_t height = 1) {

        _values.clear();

        // structure per proc data
        std::vector< std::map<index_type, value_type> >
            per_proc_data(_world.size());

        // accumulate locale values
        cb_col_itr_t col_itr;
        for(col_itr = _data.begcol(); col_itr != _data.endcol(); col_itr++) {

            // first accumulate column values, then distribute
            size_t r = 0;
            for(; r < height && _row_itr != _data.endnz(col_itr); _row_itr++, r++) {

                index_type g_cb_col_idx = col_itr.colid()  + _offset_col;
                index_type g_cb_row_idx = _row_itr.rowid() + _offset_row;
                if(_transp_b) std::swap(g_cb_row_idx, g_cb_col_idx);

                size_t target_proc = A.Owner(g_cb_row_idx, g_cb_col_idx);
                index_type coords = idx(g_cb_row_idx, g_cb_col_idx);
                per_proc_data[target_proc].insert(
                    std::make_pair(coords, _row_itr.value()));
            }
        }

        std::set<index_type> indices;
        _set_view(per_proc_data, indices);
        typename std::set<index_type>::iterator itr;
        for(itr = indices.begin(); itr != indices.end(); itr++)
            row_idxs.insert(row(*itr));
    }

    /**
     *  Extract a Elemental view of CombBLAS ROWS in sequence. The
     *  distribution of the Elemental view is determined by the distribution
     *  of the input matrix A.
     *  @Note CombBLAS is not well suited for this kind of traversals..
     */
    template<typename dist_elem_matrix_t>
    void extract_elemental_row_slab_view(
            const dist_elem_matrix_t &A,
            size_t height = 1) {

        _values.clear();

        // structure per proc data
        std::vector< std::map<index_type, value_type> >
            per_proc_data(_world.size());

        for(size_t col = 0; col < _row_itrs.size(); col++) {

            cb_nz_itr_t tmp_row_itr = _row_itrs[col];

            for(size_t i = 0; i < height && _global_row + i < _nrow; i++) {

                if(tmp_row_itr.rowid() + _offset_row != _global_row + i)
                    continue;

                index_type g_cb_col_idx = col + _offset_col;
                index_type g_cb_row_idx = tmp_row_itr.rowid() + _offset_row;
                if(_transp_b) std::swap(g_cb_row_idx, g_cb_col_idx);

                size_t target_proc = A.Owner(g_cb_row_idx, g_cb_col_idx);
                index_type coords = idx(g_cb_row_idx, g_cb_col_idx);
                per_proc_data[target_proc].insert(
                    std::make_pair(coords, tmp_row_itr.value()));

                tmp_row_itr++;
                _row_itrs[col]++;
            }
        }

        _global_row += height;

        std::set<index_type> indices;
        _set_view(per_proc_data, indices);
    }
#endif

    /**
     *  Extract the same full (redundantly stored) column view of the CombBLAS
     *  matrix on all processors.
     *  @caveat This method uses collective communication.
     */
    void extract_full_slab_view(std::set<index_type> &column_idxs,
                                const size_t slab_size = 1) {

        std::map<index_type, value_type> column_values;
        _values.clear();

        for(size_t i = 0; i < slab_size && _col_itr != _data.endcol();
            _col_itr++, i++) {

            // first accumulate column values, then distribute
            for(cb_nz_itr_t nz = _data.begnz(_col_itr);
                nz != _data.endnz(_col_itr); nz++) {

                index_type g_cb_col_idx = _col_itr.colid() + _offset_col;
                index_type g_cb_row_idx = nz.rowid()  + _offset_row;
                if(_transp_b) std::swap(g_cb_row_idx, g_cb_col_idx);

                index_type coords = idx(g_cb_row_idx, g_cb_col_idx);
                column_values.insert(std::make_pair(coords, nz.value()));
            }
        }

        std::vector< std::map< index_type, value_type> > vector_of_maps;
        boost::mpi::all_gather< std::map<index_type, value_type> > (
            _world, column_values, vector_of_maps);

        // gather vectors in one map
        typename std::map<index_type, value_type>::iterator itr;
        for(size_t i = 0; i < vector_of_maps.size(); ++i) {
            for(itr = vector_of_maps[i].begin();
                itr != vector_of_maps[i].end(); itr++) {

                column_idxs.insert(col(itr->first));

                if(_values.count(itr->first) != 0)
                    _values[itr->first] += itr->second;
                else
                    _values.insert(std::make_pair(itr->first, itr->second));
            }
        }
    }

    void extract_full_slab_view(const size_t width = 1) {

        _values.clear();

        std::map<index_type, value_type> column_values;

        for(size_t i = 0; i < width && _global_col < _ncol;
            _global_col++, i++) {

            // if I don't own anything in this column, continue
            if(_col_itr.colid() + _offset_col != _global_col)
                    continue;

            // first accumulate column values, then distribute
            for(cb_nz_itr_t nz = _data.begnz(_col_itr);
                nz != _data.endnz(_col_itr); nz++) {

                index_type g_cb_col_idx = _col_itr.colid() + _offset_col;
                index_type g_cb_row_idx = nz.rowid()  + _offset_row;
                if(_transp_b) std::swap(g_cb_row_idx, g_cb_col_idx);

                index_type coords = idx(g_cb_row_idx, g_cb_col_idx);
                column_values.insert(std::make_pair(coords, nz.value()));
            }

            // this proc can advance to next column
            _col_itr++;
        }

        std::vector< std::map< index_type, value_type> > vector_of_maps;
        boost::mpi::all_gather< std::map<index_type, value_type> > (
            _world, column_values, vector_of_maps);

        // gather vectors in one map
        typename std::map<index_type, value_type>::iterator itr;
        for(size_t i = 0; i < vector_of_maps.size(); ++i) {
            for(itr = vector_of_maps[i].begin();
                itr != vector_of_maps[i].end(); itr++) {

                if(_values.count(itr->first) != 0)
                    _values[itr->first] += itr->second;
                else
                    _values.insert(std::make_pair(itr->first, itr->second));
            }
        }
    }

    /// get the number of columns
    index_type ncols() { return _ncol; }
    index_type nrows() { return _nrow; }


private:

    /// world communicator of the view
    boost::mpi::communicator _world;

    /// the CombBLAS data
    cb_col_t &_data;

    /// the iterator holding the current position in iterating the CombBLAS
    /// matrix. This is used when generating slabs column wise.
    cb_col_itr_t _col_itr;

    /// the iterator holding the current position in iterating the CombBLAS
    /// matrix. This is used when generating slabs row wise.
    cb_nz_itr_t _row_itr;
    std::vector<cb_nz_itr_t> _row_itrs;

    /// transpose flag
    const bool _transp_b;

    /// this processors row offset in the CombBLAS matrix
    const size_t _offset_row;

    /// this processors column offset in the CombBLAS matrix
    const size_t _offset_col;

    size_t _global_row;
    size_t _global_col;

    /// number of columns
    size_t _ncol;
    /// number of rows
    size_t _nrow;

    /// holds the values of the current slab view
    std::map<index_type, value_type> _values;


    /// convert an 1D index to the global row index
    index_type row(index_type value) { return  value / _ncol; }

    /// convert an 1D index to the global column index
    index_type col(index_type value) { return  value % _ncol; }

    /// convert a (row, col) pair to a 1D index
    index_type idx(index_type row, index_type col) {
        return row * _ncol + col;
    }

    //FIXME: here we use collectives, add existing code for other comm schemes
    /// This methods takes a mapping from indices to processors and
    /// communicates the values to the target processor.
    void _set_view(
            std::vector< std::map<index_type, value_type> > &per_proc_data,
            std::set<index_type> &indices) {

        //// first we need to know how many elements we receive from other
        //// processors:
        //std::vector<size_t> n_values(_world.size());
        //for(size_t idx = 0; idx < _world.size(); ++idx)
            //n_values[idx] = per_proc_data[idx].size();

        //// resulting vector holds expected values: vector_of_sizes[from][to]
        //std::vector< std::vector<size_t> > vector_of_sizes;
        //boost::mpi::all_gather< std::vector<size_t> > (
            //_world, n_values, vector_of_sizes);

        // pre-post receives for all values then
        //FIXME: this is not working because Boost sends serialized data in
        // two steps: first the size and then the data, so waiting for one
        // irecv is not correct.
        //std::vector<boost::mpi::request> reqs;
        //std::vector< std::map<index_type, value_type> > vector_of_maps(_world.size());
        //for(size_t from = 0; from < _world.size(); ++from) {
            //size_t recv_size = vector_of_sizes[from][_world.rank()];
            //std::cout << "from " << from << " receive " << recv_size << std::endl;
            //if(recv_size > 0)
                //reqs.push_back(
                    //_world.irecv(from, 0, &vector_of_maps[from], recv_size));
        //}

        //// then perform all sends
        //for(size_t to = 0; to < _world.size(); ++to) {
            //if(per_proc_data[to].size() > 0)
                //_world.send(to, 0, per_proc_data[to]);
        //}

        //// wait for completion
        //boost::mpi::wait_all(reqs.begin(), reqs.end());

        std::vector<std::map<index_type, value_type> > vector_of_maps;
        for(int i = 0; i < _world.size(); ++i)
            boost::mpi::gather(_world, per_proc_data[i], vector_of_maps, i);

        typename std::map<index_type, value_type>::iterator itr;
        for(size_t i = 0; i < vector_of_maps.size(); ++i) {
            for(itr = vector_of_maps[i].begin();
                itr != vector_of_maps[i].end(); itr++) {

                indices.insert(itr->first);

                if(_values.count(itr->first) != 0)
                    _values[itr->first] += itr->second;
                else
                    _values.insert(std::make_pair(itr->first, itr->second));
            }
        }
    }
};

#endif

} } // namespace skylark::utility

#endif //SKYLARK_COMBBLAS_SLAB_VIEW
