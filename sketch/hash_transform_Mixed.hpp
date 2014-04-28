#ifndef SKYLARK_HASH_TRANSFORM_MIXED_HPP
#define SKYLARK_HASH_TRANSFORM_MIXED_HPP

#include <map>
#include <boost/serialization/map.hpp>

#include <CombBLAS.h>
#include <elemental.hpp>

#include "../base/exception.hpp"
#include "../base/context.hpp"

#include "../utility/external/combblas_comm_grid.hpp"

#include "transforms.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

//FIXME:
//  - Benchmark one-sided vs. col/row comm (or midpoint scheme)
//  - Most likely the scheme depends on the output Elemental distribution,
//    here we use the same comm-scheme for all of them.


/* Specialization: SpParMat for input, Elemental for output */
template <typename IndexType,
          typename ValueType,
          elem::Distribution ColDist,
          elem::Distribution RowDist,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    elem::DistMatrix<ValueType, ColDist, RowDist>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IndexType,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef SpDCCols< index_type, value_type > col_t;
    typedef FullyDistVec< index_type, value_type> mpi_vector_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    typedef elem::DistMatrix< value_type, ColDist, RowDist > output_matrix_type;
    typedef hash_transform_data_t<IndexType,
                                  ValueType,
                                  IdxDistributionType,
                                  ValueDistribution> data_type;


    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type(N, S, context) {

    }

    /**
     * Copy constructor
     */
    template <typename InputMatrixType,
              typename OutputMatrixType>
    hash_transform_t (hash_transform_t<InputMatrixType,
                                       OutputMatrixType,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (hash_transform_data_t<index_type,
                                            value_type,
                                            IdxDistributionType,
                                            ValueDistribution>& other_data) :
        data_type(other_data) {}

    template <typename Dimension>
    void apply (const matrix_type &A, output_matrix_type &sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
        } catch (std::string e) {
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg(e) );
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg(e.what()) );
        }
    }


private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply_impl (const matrix_type &A_,
        output_matrix_type &sketch_of_A,
        Dimension dist) const {

        // We are essentially doing a 'const' access to A, but the necessary,
        // 'const' option is missing from the interface
        matrix_type &A = const_cast<matrix_type&>(A_);

        const size_t rank = A.getcommgrid()->GetRank();

        // extract columns of matrix
        col_t &data = A.seq();

        const size_t ncols = sketch_of_A.Width();
        const size_t nrows = sketch_of_A.Height();

        const size_t my_row_offset = utility::cb_my_row_offset(A);
        const size_t my_col_offset = utility::cb_my_col_offset(A);

        size_t comm_size = A.getcommgrid()->GetSize();
        std::vector< std::set<size_t> > proc_set(comm_size);

        // pre-compute processor targets of local sketch application
        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                // compute global row and column id, and compress in one
                // target position index
                const index_type rowid = nz.rowid()  + my_row_offset;
                const index_type colid = col.colid() + my_col_offset;
                const size_t pos       = getPos(rowid, colid, ncols, dist);

                // compute target processor for this target index
                const size_t target = sketch_of_A.Owner(pos / ncols, pos % ncols);

                if(proc_set[target].count(pos) == 0) {
                    assert(target < comm_size);
                    proc_set[target].insert(pos);
                }
            }
        }

        // constructing arrays for one-sided access
        std::vector<size_t>     proc_size(comm_size, 0);
        std::vector<index_type> proc_start_idx(comm_size, 0);
        proc_size[0] = proc_set[0].size();
        for(size_t i = 1; i < proc_start_idx.size(); ++i) {
            proc_size[i]      = proc_set[i].size();
            proc_start_idx[i] = proc_start_idx[i-1] + proc_size[i-1];
        }

        size_t nnz = proc_start_idx[comm_size-1] + proc_size[comm_size-1];
        std::vector<index_type> indicies(nnz, 0);
        std::vector<value_type> values(nnz, 0);

        // Apply sketch for all local values. Note that some of the resulting
        // values might end up on a different processor. The data structure
        // fills values (sorted by processor id) in one continuous array.
        // Subsequently, one-sided operations can be used to access values for
        // each processor.
        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                // compute global row and column id, and compress in one
                // target position index
                const index_type rowid = nz.rowid()  + my_row_offset;
                const index_type colid = col.colid() + my_col_offset;
                const size_t pos       = getPos(rowid, colid, ncols, dist);

                // compute target processor for this target index
                const size_t proc = sketch_of_A.Owner(pos / ncols, pos % ncols);

                // get offset in array for current element
                const size_t ar_idx = proc_start_idx[proc] +
                    std::distance(proc_set[proc].begin(), proc_set[proc].find(pos));

                indicies[ar_idx] = pos;
                values[ar_idx]  += nz.value() * getRowValue(rowid, colid, dist);
            }
        }

        // Creating windows for all relevant arrays
        ///FIXME: MPI-3 stuff?
        boost::mpi::communicator comm = utility::get_communicator(A);
        MPI_Win proc_win, start_offset_win, idx_win, val_win;
        MPI_Win_create(&proc_size[0], sizeof(size_t) * comm_size,
                       sizeof(size_t), MPI_INFO_NULL, comm, &proc_win);

        MPI_Win_create(&proc_start_idx[0], sizeof(size_t) * comm_size,
                       sizeof(size_t), MPI_INFO_NULL, comm, &start_offset_win);

        MPI_Win_create(&indicies[0], sizeof(index_type) * indicies.size(),
                       sizeof(index_type), MPI_INFO_NULL, comm, &idx_win);

        MPI_Win_create(&values[0], sizeof(value_type) * values.size(),
                       sizeof(value_type), MPI_INFO_NULL, comm, &val_win);

        MPI_Win_fence(0, proc_win);
        MPI_Win_fence(0, start_offset_win);
        MPI_Win_fence(0, idx_win);
        MPI_Win_fence(0, val_win);


        // accumulate values from other procs
        std::map<size_t, value_type> vals_map;
        for(size_t p = 0; p < comm_size; ++p) {

            // since all procs need to call the fence we gather all the
            // necessary values
            size_t num_values = 0;
            MPI_Get(&num_values, 1, boost::mpi::get_mpi_datatype<size_t>(),
                    p, rank, 1, boost::mpi::get_mpi_datatype<size_t>(),
                    proc_win);
            MPI_Win_fence(0, proc_win);

            size_t offset = 0;
            MPI_Get(&offset, 1, boost::mpi::get_mpi_datatype<size_t>(),
                    p, rank, 1, boost::mpi::get_mpi_datatype<size_t>(),
                    start_offset_win);
            MPI_Win_fence(0, start_offset_win);

            // since all procs need to call the fence we fill indices and
            // values even if num_values can be 0 (= don't get data).
            std::vector<index_type> add_idx(num_values);
            std::vector<value_type> add_val(num_values);
            MPI_Get(&(add_idx[0]), num_values,
                    boost::mpi::get_mpi_datatype<index_type>(), p, offset,
                    num_values, boost::mpi::get_mpi_datatype<index_type>(),
                    idx_win);
            MPI_Win_fence(0, idx_win);

            MPI_Get(&(add_val[0]), num_values,
                    boost::mpi::get_mpi_datatype<value_type>(), p, offset,
                    num_values, boost::mpi::get_mpi_datatype<value_type>(),
                    val_win);
            MPI_Win_fence(0, val_win);

            // finally, add data to local buffer (if we have any).
            for(size_t i = 0; i < num_values; ++i) {
                if(vals_map.count(add_idx[i]) != 0)
                    vals_map[add_idx[i]] += add_val[i];
                else
                    vals_map.insert(std::make_pair(add_idx[i], add_val[i]));
            }
        }

        // fill into sketch matrix
        typename std::map<size_t, value_type>::const_iterator itr;
        for(itr = vals_map.begin(); itr != vals_map.end(); itr++) {
            index_type lrow = sketch_of_A.LocalRow(itr->first / ncols);
            index_type lcol = sketch_of_A.LocalCol(itr->first % ncols);
            value_type val  = itr->second;
            sketch_of_A.SetLocal(lrow, lcol, val);
        }

        MPI_Win_fence(0, proc_win);
        MPI_Win_fence(0, start_offset_win);
        MPI_Win_fence(0, idx_win);
        MPI_Win_fence(0, val_win);

        MPI_Win_free(&proc_win);
        MPI_Win_free(&start_offset_win);
        MPI_Win_free(&idx_win);
        MPI_Win_free(&val_win);
    }


    inline index_type getPos(index_type rowid, index_type colid, size_t ncols,
        columnwise_tag) const {
        return colid + ncols * data_type::row_idx[rowid];
    }

    inline index_type getPos(index_type rowid, index_type colid, size_t ncols,
        rowwise_tag) const {
        return rowid * ncols + data_type::row_idx[colid];
    }

    inline value_type getRowValue(index_type rowid, index_type colid,
        columnwise_tag) const {
        return data_type::row_value[rowid];
    }

    inline value_type getRowValue(index_type rowid, index_type colid,
        rowwise_tag) const {
        return data_type::row_value[colid];
    }
};


/* Specialization: SpParMat for input, Local Elemental output */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    elem::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IndexType,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef SpDCCols< index_type, value_type > col_t;
    typedef FullyDistVec< index_type, value_type> mpi_vector_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    typedef elem::Matrix< value_type > output_matrix_type;
    typedef hash_transform_data_t<IndexType,
                                  ValueType,
                                  IdxDistributionType,
                                  ValueDistribution> data_type;


    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type(N, S, context) {

    }

    /**
     * Copy constructor
     */
    template <typename InputMatrixType,
              typename OutputMatrixType>
    hash_transform_t (hash_transform_t<InputMatrixType,
                                       OutputMatrixType,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    hash_transform_t (hash_transform_data_t<index_type,
                                            value_type,
                                            IdxDistributionType,
                                            ValueDistribution>& other_data) :
        data_type(other_data) {}

    template <typename Dimension>
    void apply (const matrix_type &A, output_matrix_type &sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
        } catch (std::string e) {
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg(e) );
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                base::combblas_exception()
                    << base::error_msg(e.what()) );
        }
    }


private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply_impl (const matrix_type &A_,
        output_matrix_type &sketch_of_A,
        Dimension dist) const {

        // We are essentially doing a 'const' access to A, but the necessary,
        // 'const' option is missing from the interface
        matrix_type &A = const_cast<matrix_type&>(A_);

        const size_t rank = A.getcommgrid()->GetRank();

        // extract columns of matrix
        col_t &data = A.seq();

        const size_t my_row_offset = utility::cb_my_row_offset(A);
        const size_t my_col_offset = utility::cb_my_col_offset(A);

        int n_res_cols = A.getncol();
        int n_res_rows = A.getnrow();
        get_res_size(n_res_rows, n_res_cols, dist);

        // Apply sketch for all local values. Subsequently, all values are
        // gathered on processor 0 and the local matrix is populated.
        typedef std::map<index_type, value_type> col_values_t;
        col_values_t col_values;
        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                index_type rowid = nz.rowid()  + my_row_offset;
                index_type colid = col.colid() + my_col_offset;

                const value_type value =
                    nz.value() * getValue(rowid, colid, dist);
                finalPos(rowid, colid, dist);
                col_values[colid * n_res_rows + rowid] += value;
            }
        }

        std::vector< std::map<index_type, value_type > >
            result;
        boost::mpi::gather(utility::get_communicator(A), col_values, result, 0);

        if(rank == 0) {
            typedef typename std::map<index_type, value_type>::iterator itr_t;
            for(size_t i = 0; i < result.size(); ++i) {
                itr_t proc_itr = result[i].begin();
                for(; proc_itr != result[i].end(); proc_itr++) {
                    int row = proc_itr->first % n_res_rows;
                    int col = proc_itr->first / n_res_rows;
                    sketch_of_A.Update(row, col, proc_itr->second);
                }
            }
        }
    }

    inline void finalPos(index_type &rowid, index_type &colid,
                         columnwise_tag) const {
        rowid = data_type::row_idx[rowid];
    }

    inline void finalPos(index_type &rowid, index_type &colid,
                         rowwise_tag) const {
        colid = data_type::row_idx[colid];
    }

    inline value_type getValue(index_type rowid, index_type colid,
                               columnwise_tag) const {
        return data_type::row_value[rowid];
    }

    inline value_type getValue(index_type rowid, index_type colid,
                               rowwise_tag) const {
        return data_type::row_value[colid];
    }

    inline void get_res_size(int &rows, int &cols, columnwise_tag) const {
        rows = data_type::_S;
    }

    inline void get_res_size(int &rows, int &cols, rowwise_tag) const {
        cols = data_type::_S;
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_MIXED_HPP
