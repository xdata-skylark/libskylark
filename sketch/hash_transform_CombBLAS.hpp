#ifndef SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP
#define SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP

#include <map>
#include "boost/serialization/map.hpp"

#include <CombBLAS.h>
#include "../utility/external/FullyDistMultiVec.hpp"
#include "../utility/exception.hpp"

#include "context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

/* Specialization: FullyDistMultiVec for input, output */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <FullyDistMultiVec<IndexType, ValueType>,
                         FullyDistMultiVec<IndexType, ValueType>,
                         IdxDistributionType,
                         ValueDistribution > :
        public hash_transform_data_t<IndexType,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef FullyDistMultiVec<IndexType, ValueType> matrix_type;
    typedef FullyDistMultiVec<IndexType, ValueType> output_matrix_type;
    typedef FullyDistVec<IndexType, ValueType> mpi_vector_t;
    typedef FullyDistMultiVec<IndexType, ValueType> mpi_multi_vector_t;
    typedef hash_transform_data_t<IndexType,
                                  ValueType,
                                  IdxDistributionType,
                                  ValueDistribution> base_data_t;

    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, skylark::sketch::context_t& context) :
        base_data_t (N, S, context) {}

    /**
     * Copy constructor
     */
    template <typename InputMatrixType,
              typename OutputMatrixType>
    hash_transform_t (hash_transform_t<InputMatrixType,
                                       OutputMatrixType,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    /**
     * Constructor from data
     */
    hash_transform_t (hash_transform_data_t<index_type,
                                            value_type,
                                            IdxDistributionType,
                                            ValueDistribution>& other_data) :
        base_data_t(other_data.get_data()) {}

    template <typename Dimension>
    void apply (const mpi_multi_vector_t &A,
        mpi_multi_vector_t &sketch_of_A,
        Dimension dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        } catch (std::string e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e) );
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e.what()) );
        }
    }


private:
    void apply_impl_single (const mpi_vector_t& a_,
        mpi_vector_t& sketch_of_a,
        columnwise_tag) const {
        std::vector<value_type> sketch_term(base_data_t::S,0);

        // We are essentially doing a 'const' access to a, but the neccessary,
        // 'const' option is missing from the interface.
        mpi_vector_t &a = const_cast<mpi_vector_t&>(a_);

        /** Accumulate the local sketch vector */
        /** FIXME: Lot's of random access --- not good for performance */
        DenseVectorLocalIterator<index_type, value_type> local_iter(a);
        while(local_iter.HasNext()) {
            index_type idx = local_iter.GetLocIndex();
            index_type global_idx = local_iter.LocalToGlobal(idx);
            index_type global_sketch_idx = base_data_t::row_idx[global_idx];
            sketch_term[global_sketch_idx] +=
                (local_iter.GetValue()*base_data_t::row_value[global_idx]);
            local_iter.Next();
        }

        /** Accumulate the global sketch vector */
        /** FIXME: Only need to scatter ... don't need everything everywhere */
        MPI_Allreduce(MPI_IN_PLACE,
            &(sketch_term[0]),
            base_data_t::S,
            MPIType<value_type>(),
            MPI_SUM,
            a.commGrid->GetWorld());

        /** Fill in .. SetElement() is dummy for non-local sets, so it's ok */
        for (index_type i=0; i<base_data_t::S; ++i) {
            sketch_of_a.SetElement(i,sketch_term[i]);
        }
    }


    void apply_impl (const mpi_multi_vector_t& A,
        mpi_multi_vector_t& sketch_of_A,
        columnwise_tag) const {
        const index_type num_rhs = A.size;
        if (sketch_of_A.size != num_rhs) { /** error */; return; }
        if (A.dim != base_data_t::N) { /** error */; return; }
        if (sketch_of_A.dim != base_data_t::S) { /** error */; return; }

        /** FIXME: Can sketch all the vectors in one shot */
        for (index_type i=0; i<num_rhs; ++i) {
            apply_impl_single (A[i], sketch_of_A[i], columnwise_tag());
        }
    }
};


/* Specialization: SpParMat for input, output */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IndexType,
                                     ValueType,
                                     IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef SpDCCols< IndexType, value_type > col_t;
    typedef FullyDistVec< IndexType, ValueType > mpi_vector_t;
    typedef SpParMat< IndexType, value_type, col_t > matrix_type;
    typedef SpParMat< IndexType, value_type, col_t > output_matrix_type;
    typedef hash_transform_data_t<IndexType,
                                  ValueType,
                                  IdxDistributionType,
                                  ValueDistribution> base_data_t;
    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, skylark::sketch::context_t& context) :
        base_data_t(N, S, context) {}

    /**
     * Copy constructor
     */
    template <typename InputMatrixType,
              typename OutputMatrixType>
    hash_transform_t (hash_transform_t<InputMatrixType,
                                       OutputMatrixType,
                                       IdxDistributionType,
                                       ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    /**
     * Constructor from data
     */
    hash_transform_t (hash_transform_data_t<index_type,
                                            value_type,
                                            IdxDistributionType,
                                            ValueDistribution>& other_data) :
        base_data_t(other_data.get_data()) {}

    template <typename Dimension>
    void apply (const matrix_type &A,
        output_matrix_type &sketch_of_A,
        Dimension dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                utility::mpi_exception()
                    << utility::error_msg(e.what()) );
        } catch (std::string e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e) );
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                utility::combblas_exception()
                    << utility::error_msg(e.what()) );
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

        // We are essentially doing a 'const' access to A, but the neccessary,
        // 'const' option is missing from the interface
        matrix_type &A = const_cast<matrix_type&>(A_);

        const size_t rank = A.getcommgrid()->GetRank();

        // extract columns of matrix
        col_t &data = A.seq();

        const size_t ncols = sketch_of_A.getncol();
        const size_t nrows = sketch_of_A.getnrow();

        //FIXME: use comm_grid
        const size_t my_row_offset =
            static_cast<int>(0.5 + (static_cast<double>(A.getnrow()) /
                    A.getcommgrid()->GetGridRows())) *
            A.getcommgrid()->GetRankInProcCol(rank);

        const size_t my_col_offset =
            static_cast<int>(0.5 + (static_cast<double>(A.getncol()) /
                    A.getcommgrid()->GetGridCols())) *
            A.getcommgrid()->GetRankInProcRow(rank);

        // Pre-compute processor targets
        std::vector<index_type> proc_seq;
        std::vector<index_type> proc_size(A.getcommgrid()->GetSize(), 0);
        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                const index_type rowid = nz.rowid()  + my_row_offset;
                const index_type colid = col.colid() + my_col_offset;

                const size_t target_proc = compute_proc(A, rowid, colid, dist);
                proc_seq.push_back(target_proc);
                proc_size[target_proc]++;
            }
        }

        // constructing arrays for one-sided access
        size_t idx = 0;
        const size_t total_nz = proc_seq.size();
        std::vector<index_type> proc_start_idx(A.getcommgrid()->GetSize(), 0);
        for(size_t i = 1; i < proc_start_idx.size(); ++i)
           proc_start_idx[i] = proc_start_idx[i-1] + proc_size[i-1];

        std::vector<index_type> indicies(total_nz, 0);
        std::vector<value_type> values(total_nz, 0);

        // Apply sketch for all local values. Note that some of the resulting
        // values might end up on a different processor. The datastructure
        // fills values (sorted by processor id) in one continuous array.
        // Subsequently one-sided operations can be used to gather values for
        // each processor.
        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                const index_type rowid = nz.rowid()  + my_row_offset;
                const index_type colid = col.colid() + my_col_offset;

                // get offset in array for current element
                const size_t ar_idx = proc_start_idx[proc_seq[idx]]++;

                indicies[ar_idx] = getPos(rowid, colid, ncols, dist);
                values[ar_idx]   = nz.value() * getRowValue(rowid, colid, dist);

                idx++;
            }
        }

        // Creating windows for all relevant arrays
        ///FIXME: MPI-3 stuff?
        MPI_Win proc_win, idx_win, val_win;
        MPI_Win_create(&(proc_size[0]), sizeof(size_t) * proc_size.size(), 1,
                       MPI_INFO_NULL, A.getcommgrid()->GetWorld(), &proc_win);

        MPI_Win_create(&(indicies[0]), sizeof(index_type) * indicies.size(), 1,
                       MPI_INFO_NULL, A.getcommgrid()->GetWorld(), &idx_win);

        MPI_Win_create(&(values[0]), sizeof(value_type) * values.size(), 1,
                       MPI_INFO_NULL, A.getcommgrid()->GetWorld(), &val_win);

        MPI_Win_fence(MPI_MODE_NOPUT, proc_win);
        MPI_Win_fence(MPI_MODE_NOPUT, idx_win);
        MPI_Win_fence(MPI_MODE_NOPUT, val_win);


        // get values from other procs
        std::map<size_t, value_type> vals_map;

        for(size_t p = 0; p < A.getcommgrid()->GetSize(); ++p) {

            size_t num_values = 0;
            MPI_Get(&num_values, 1, MPI_INT, p, rank, 1, MPI_INT, proc_win);

            //FIXME: MPI types
            std::vector<size_t> add_idx;
            std::vector<value_type> add_val;
            MPI_Get(&(add_idx[0]), num_values, MPI_INT, p, rank,
                    num_values, MPI_INT, idx_win);
            MPI_Get(&(add_val[0]), num_values, MPI_DOUBLE, p, rank,
                    num_values, MPI_DOUBLE, val_win);

            for(size_t i = 0; i < num_values; ++i) {
                if(vals_map.count(add_idx[i]) != 0)
                    vals_map[add_idx[i]] += add_val[i];
                else
                    vals_map.insert(std::make_pair(add_idx[i], add_val[i]));
            }
        }

        // .. and finally create a new sparse matrix
        const size_t matrix_size = vals_map.size();
        mpi_vector_t cols(matrix_size);
        mpi_vector_t rows(matrix_size);
        mpi_vector_t vals(matrix_size);
        idx = 0;

        typename std::map<size_t, value_type>::const_iterator itr;
        for(itr = vals_map.begin(); itr != vals_map.end(); itr++, idx++) {
            cols.SetElement(idx, itr->first % ncols);
            rows.SetElement(idx, itr->first / ncols);
            vals.SetElement(idx, itr->second);
        }

        //FIXME: can we set sketch_of_A directly? (See SparseCommon, Owner)
        output_matrix_type tmp(sketch_of_A.getnrow(), sketch_of_A.getncol(),
            rows, cols, vals);

        //delete sketch_of_A.spSeq;
        sketch_of_A = tmp;

        //FIXME: add a method for SpParMat to allow setting rows/cols/vals
        //       directly..
        // and fill into sketch matrix
        //vector< vector < tuple<index_type, index_type, value_type> > >
        // data_val (
        //rows.commGrid->GetSize() );

        //index_type locsize = rows.LocArrSize();
        //for(index_type i = 0; i < locsize; ++i) {
        //index_type lrow, lcol;
        //int owner = sketch_of_A.Owner(sketch_of_A.getnrow(),
        //sketch_of_A.getncol(),
        //rows[i], cols[i], lrow, lcol);
        //data_val[owner].push_back(make_tuple(lrow, lcol, vals[i]));
        //}
        //sketch_of_A.SparseCommon(data_val, locsize, sketch_of_A.getnrow(),
                             //sketch_of_A.getncol());

        MPI_Win_free(&proc_win);
        MPI_Win_free(&idx_win);
        MPI_Win_free(&val_win);
    }


    inline index_type getPos(index_type rowid, index_type colid, size_t ncols,
        columnwise_tag) const {
        return colid + ncols * base_data_t::row_idx[rowid];
    }

    inline index_type getPos(index_type rowid, index_type colid, size_t ncols,
        rowwise_tag) const {
        return rowid * ncols + base_data_t::row_idx[colid];
    }

    inline value_type getRowValue(index_type rowid, index_type colid,
        columnwise_tag) const {
        return base_data_t::row_value[rowid];
    }

    inline value_type getRowValue(index_type rowid, index_type colid,
        rowwise_tag) const {
        return base_data_t::row_value[colid];
    }

    //FIXME: move to comm_grid
    inline size_t compute_proc(const matrix_type &A, const index_type row,
                               const index_type col, columnwise_tag) const {

        const index_type trow = base_data_t::row_idx[row];
        const size_t rows_per_proc = static_cast<size_t>(
            (static_cast<double>(A.getnrow()) / A.getcommgrid()->GetGridRows()));

        return A.getcommgrid()->GetRank(
            static_cast<size_t>(trow / rows_per_proc),
            A.getcommgrid()->GetRankInProcCol());
    }

    //FIXME: move to comm_grid
    inline size_t compute_proc(const matrix_type &A, const index_type row,
                               const index_type col, rowwise_tag) const {

        const index_type tcol = base_data_t::row_idx[col];
        const size_t cols_per_proc = static_cast<size_t>(
            (static_cast<double>(A.getncol()) / A.getcommgrid()->GetGridCols()));

        return A.getcommgrid()->GetRank(
            A.getcommgrid()->GetRankInProcRow(),
            static_cast<size_t>(tcol / cols_per_proc));
    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP
