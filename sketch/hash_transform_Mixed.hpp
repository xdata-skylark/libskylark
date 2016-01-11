#ifndef SKYLARK_HASH_TRANSFORM_MIXED_HPP
#define SKYLARK_HASH_TRANSFORM_MIXED_HPP

#include <map>
#include <boost/serialization/map.hpp>

#include "../base/sparse_vc_star_matrix.hpp"

#include "../utility/external/combblas_comm_grid.hpp"
#include "../utility/external/elemental_comm_grid.hpp"

namespace skylark { namespace sketch {

//FIXME:
//  - Benchmark one-sided vs. col/row comm (or midpoint scheme):
//    Most likely the scheme depends on the output Elemental distribution,
//    here we use the same comm-scheme for all output types.
//  - Processing Sparse matrix in blocks?
//  - MPI-3 stuff, see: Enabling highly-scalable remote memory access
//    programming with MPI-3 one sided, R. Gerstenbergerm,  M. Besta, and
//    T. Hoefler.


/* Specialization: sparse_vc_star for input, distributed Elemental for output */
template <typename ValueType,
          El::Distribution ColDist,
          El::Distribution RowDist,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    base::sparse_vc_star_matrix_t<ValueType>,
    El::DistMatrix<ValueType, ColDist, RowDist>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {
    typedef El::Int index_type;
    typedef ValueType value_type;
    typedef base::sparse_vc_star_matrix_t<value_type> matrix_type;
    typedef El::DistMatrix< value_type, ColDist, RowDist > output_matrix_type;
    typedef hash_transform_data_t<IdxDistributionType,
                                  ValueDistribution> data_type;


    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context)
        : data_type(N, S, context)
    {}

    /**
     * Copy constructor
     */
    hash_transform_t (
        hash_transform_t<
            matrix_type, output_matrix_type,
            IdxDistributionType, ValueDistribution>& other)
        : data_type(other)
    {}

    /**
     * Constructor from data
     */
    hash_transform_t (const data_type& other_data)
        : data_type(other_data)
    {}

    template <typename Dimension>
    void apply (const matrix_type &A, output_matrix_type &sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl (A, sketch_of_A, dimension);
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
        } catch (std::bad_alloc e) {
            SKYLARK_THROW_EXCEPTION (
                base::skylark_exception()
                    << base::error_msg("bad_alloc: out of memory") );
        }
    }


private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     *
     * FIXME: distribution depending schemes would be more efficient
     */
    template <typename Dimension>
    void apply_impl (const matrix_type &A,
        output_matrix_type &sketch_of_A,
        Dimension dist) const {

        typedef size_t offset_idx_t;

        boost::mpi::communicator comm = skylark::utility::get_communicator(A);

        const size_t rank = comm.rank();

        const size_t ncols = sketch_of_A.Width();

        size_t comm_size = comm.size();

        const int* A_indptr  = A.indptr();
        const int* A_indices = A.indices();
        const value_type *A_values = A.locked_values();

        std::vector< std::map<size_t, size_t> > array_offsets(comm_size);

        // pre-compute processor targets of local sketch application
        for(int i = 0; i < A.width(); i++) {
            for (int j = A_indptr[i]; j < A_indptr[i + 1]; j++) {

                // compute global row and column id, and compress in one
                // target position index
                const size_t pos = getPos(
                        A.global_row(A_indices[j]), i, ncols, dist);

                // compute target processor for this target index
                const size_t target_rank = utility::owner(
                        sketch_of_A, pos / ncols, pos % ncols);

                assert(target_rank < comm_size);

                // map the position to the next empty slot
                if(array_offsets[target_rank].count(pos) == 0) {
                     size_t next_pos = array_offsets[target_rank].size();
                     array_offsets[target_rank][pos] = next_pos;
                }
            }
        }


        // constructing array holding start/end indices for one-sided access
        std::vector<offset_idx_t> proc_start_idx(comm_size + 1, 0);
        for(size_t i = 1; i < comm_size + 1; ++i)
            proc_start_idx[i] = proc_start_idx[i-1] + array_offsets[i-1].size();

        const size_t my_num_values = proc_start_idx[comm_size];
        // total number of nnz that will result when applying sketch locally
        std::vector<index_type> indices(my_num_values, 0);
        std::vector<value_type> values(my_num_values, 0);

        // Apply sketch for all local values. Note that some of the resulting
        // values might end up on a different processor. The data structure
        // fills values (sorted by processor id) in one continuous array.
        // Subsequently, one-sided operations can be used to access values for
        // each processor.
        for(int i = 0; i < A.width(); i++) {
            for (int j = A_indptr[i]; j < A_indptr[i + 1]; j++) {

                // compute global row and column id, and compress in one
                // target position index
                const size_t pos = getPos(
                        A.global_row(A_indices[j]), i, ncols, dist);

                // compute target processor for this target index
                const size_t target_rank = utility::owner(
                        sketch_of_A, pos / ncols, pos % ncols);

                assert(target_rank < comm_size);

                // get offset in array for current element
                const size_t ar_idx = proc_start_idx[target_rank] +
                    array_offsets[target_rank][pos];

                assert(ar_idx < indices.size());
                indices[ar_idx] = pos;

                assert(ar_idx < values.size());
                values[ar_idx]  += A_values[j] *
                    data_type::getValue(A_indices[j], i, dist);
            }
        }

        // tell MPI that we will not use locks
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "no_locks", "true");

        MPI_Win start_offset_win, idx_win, val_win;

        MPI_Win_create(&proc_start_idx[0], sizeof(size_t) * (comm_size + 1),
                       sizeof(size_t), info, comm, &start_offset_win);

        MPI_Win_create(&indices[0], sizeof(index_type) * indices.size(),
                       sizeof(index_type), info, comm, &idx_win);

        MPI_Win_create(&values[0], sizeof(value_type) * values.size(),
                       sizeof(value_type), info, comm, &val_win);

        MPI_Info_free(&info);

        // Synchronize epoch, no subsequent put operations (read only) and no
        // preceding fence calls.
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, start_offset_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, idx_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, val_win);


        // accumulate values from other procs
        for(size_t p = 0; p < comm_size; ++p) {

            // get the start/end offset
            std::vector<size_t> offset(2);
            MPI_Get(&(offset[0]), 2, boost::mpi::get_mpi_datatype<size_t>(),
                    p, rank, 2, boost::mpi::get_mpi_datatype<size_t>(),
                    start_offset_win);

            MPI_Win_fence(MPI_MODE_NOPUT, start_offset_win);
            size_t num_values = offset[1] - offset[0];

            // and fill indices/values.
            std::vector<index_type> add_idx(num_values);
            std::vector<value_type> add_val(num_values);
            MPI_Get(&(add_idx[0]), num_values,
                    boost::mpi::get_mpi_datatype<index_type>(), p, offset[0],
                    num_values, boost::mpi::get_mpi_datatype<index_type>(),
                    idx_win);

            MPI_Get(&(add_val[0]), num_values,
                    boost::mpi::get_mpi_datatype<value_type>(), p, offset[0],
                    num_values, boost::mpi::get_mpi_datatype<value_type>(),
                    val_win);

            MPI_Win_fence(MPI_MODE_NOPUT, idx_win);
            MPI_Win_fence(MPI_MODE_NOPUT, val_win);

            // finally, set data in local buffer
            for(size_t i = 0; i < num_values; ++i) {
                index_type lrow = sketch_of_A.LocalRow(add_idx[i] / ncols);
                index_type lcol = sketch_of_A.LocalCol(add_idx[i] % ncols);
                sketch_of_A.UpdateLocal(lrow, lcol, add_val[i]);
            }
        }

        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, start_offset_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, idx_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, val_win);

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
};

#if SKYLARK_HAVE_COMBBLAS
/* Specialization: SpParMat for input, distributed Elemental for output */
template <typename IndexType,
          typename ValueType,
          El::Distribution ColDist,
          El::Distribution RowDist,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    El::DistMatrix<ValueType, ColDist, RowDist>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef SpDCCols< index_type, value_type > col_t;
    typedef FullyDistVec< index_type, value_type> mpi_vector_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    typedef El::DistMatrix< value_type, ColDist, RowDist > output_matrix_type;
    typedef hash_transform_data_t<IdxDistributionType,
                                  ValueDistribution> data_type;


    /**
     * Regular constructor
     */
    hash_transform_t (int N, int S, base::context_t& context) :
        data_type(N, S, context)
    {}

    /**
     * Copy constructor
     */
    hash_transform_t (
        hash_transform_t<
            matrix_type, output_matrix_type,
            IdxDistributionType, ValueDistribution>& other)
        : data_type(other)
    {}

    /**
     * Constructor from data
     */
    hash_transform_t (const data_type& other_data)
        : data_type(other_data)
    {}

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
        } catch (std::bad_alloc e) {
            SKYLARK_THROW_EXCEPTION (
                base::skylark_exception()
                    << base::error_msg("bad_alloc: out of memory") );
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
                const size_t target = utility::owner(
                        sketch_of_A, pos / ncols, pos % ncols);

                if(proc_set[target].count(pos) == 0) {
                    assert(target < comm_size);
                    proc_set[target].insert(pos);
                }
            }
        }

        // constructing array holding start/end indices for one-sided access
        std::vector<index_type> proc_start_idx(comm_size + 1, 0);
        for(size_t i = 1; i < comm_size + 1; ++i)
            proc_start_idx[i] = proc_start_idx[i-1] + proc_set[i-1].size();

        // total number of nnz that will result when applying sketch locally
        std::vector<index_type> indicies(proc_start_idx[comm_size], 0);
        std::vector<value_type> values(proc_start_idx[comm_size], 0);

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
                const size_t proc = utility::owner(
                        sketch_of_A, pos / ncols, pos % ncols);

                // get offset in array for current element
                const size_t ar_idx = proc_start_idx[proc] +
                    std::distance(proc_set[proc].begin(), proc_set[proc].find(pos));

                indicies[ar_idx] = pos;
                values[ar_idx]  += nz.value() *
                                   data_type::getValue(rowid, colid, dist);
            }
        }

        // Creating windows for all relevant arrays
        boost::mpi::communicator comm = utility::get_communicator(A);

        // tell MPI that we will not use locks
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "no_locks", "true");

        MPI_Win start_offset_win, idx_win, val_win;

        MPI_Win_create(&proc_start_idx[0], sizeof(size_t) * (comm_size + 1),
                       sizeof(size_t), info, comm, &start_offset_win);

        MPI_Win_create(&indicies[0], sizeof(index_type) * indicies.size(),
                       sizeof(index_type), info, comm, &idx_win);

        MPI_Win_create(&values[0], sizeof(value_type) * values.size(),
                       sizeof(value_type), info, comm, &val_win);

        MPI_Info_free(&info);

        // Synchronize epoch, no subsequent put operations (read only) and no
        // preceding fence calls.
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, start_offset_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, idx_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, val_win);


        // accumulate values from other procs
        for(size_t p = 0; p < comm_size; ++p) {

            // get the start/end offset
            std::vector<size_t> offset(2);
            MPI_Get(&(offset[0]), 2, boost::mpi::get_mpi_datatype<size_t>(),
                    p, rank, 2, boost::mpi::get_mpi_datatype<size_t>(),
                    start_offset_win);

            MPI_Win_fence(MPI_MODE_NOPUT, start_offset_win);
            size_t num_values = offset[1] - offset[0];

            // and fill indices/values.
            std::vector<index_type> add_idx(num_values);
            std::vector<value_type> add_val(num_values);
            MPI_Get(&(add_idx[0]), num_values,
                    boost::mpi::get_mpi_datatype<index_type>(), p, offset[0],
                    num_values, boost::mpi::get_mpi_datatype<index_type>(),
                    idx_win);

            MPI_Get(&(add_val[0]), num_values,
                    boost::mpi::get_mpi_datatype<value_type>(), p, offset[0],
                    num_values, boost::mpi::get_mpi_datatype<value_type>(),
                    val_win);

            MPI_Win_fence(MPI_MODE_NOPUT, idx_win);
            MPI_Win_fence(MPI_MODE_NOPUT, val_win);

            // finally, set data in local buffer
            for(size_t i = 0; i < num_values; ++i) {
                index_type lrow = sketch_of_A.LocalRow(add_idx[i] / ncols);
                index_type lcol = sketch_of_A.LocalCol(add_idx[i] % ncols);
                sketch_of_A.UpdateLocal(lrow, lcol, add_val[i]);
            }
        }

        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, start_offset_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, idx_win);
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, val_win);

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
};

/* Specialization: SpParMat for input, Local Elemental output */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    El::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef SpDCCols< index_type, value_type > col_t;
    typedef FullyDistVec< index_type, value_type> mpi_vector_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    typedef El::Matrix< value_type > output_matrix_type;
    typedef hash_transform_data_t<IdxDistributionType,
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
    hash_transform_t (hash_transform_data_t<IdxDistributionType,
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
        } catch (std::bad_alloc e) {
            SKYLARK_THROW_EXCEPTION (
                base::skylark_exception()
                    << base::error_msg("bad_alloc: out of memory") );
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
        data_type::get_res_size(n_res_rows, n_res_cols, dist);

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
                    nz.value() * data_type::getValue(rowid, colid, dist);
                data_type::finalPos(rowid, colid, dist);
                col_values[colid * n_res_rows + rowid] += value;
            }
        }

        std::vector< std::map<index_type, value_type > > result;
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
};


/* Specialization: SpParMat for input, Elemental[* / *] output */
template <typename IndexType,
          typename ValueType,
          template <typename> class IdxDistributionType,
          template <typename> class ValueDistribution>
struct hash_transform_t <
    SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    IdxDistributionType,
    ValueDistribution > :
        public hash_transform_data_t<IdxDistributionType,
                                     ValueDistribution> {
    typedef IndexType index_type;
    typedef ValueType value_type;
    typedef SpDCCols< index_type, value_type > col_t;
    typedef FullyDistVec< index_type, value_type> mpi_vector_t;
    typedef SpParMat< index_type, value_type, col_t > matrix_type;
    typedef El::DistMatrix< value_type, El::STAR, El::STAR > output_matrix_type;
    typedef hash_transform_data_t<IdxDistributionType,
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
    hash_transform_t (hash_transform_data_t<IdxDistributionType,
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
        } catch (std::bad_alloc e) {
            SKYLARK_THROW_EXCEPTION (
                base::skylark_exception()
                    << base::error_msg("bad_alloc: out of memory") );
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
        data_type::get_res_size(n_res_rows, n_res_cols, dist);

        // Apply sketch for all local values. Subsequently, all values are
        // gathered on all processor and the "local" matrix is populated.
        typedef std::map<index_type, value_type> col_values_t;
        col_values_t col_values;
        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                index_type rowid = nz.rowid()  + my_row_offset;
                index_type colid = col.colid() + my_col_offset;

                const value_type value =
                    nz.value() * data_type::getValue(rowid, colid, dist);
                data_type::finalPos(rowid, colid, dist);
                col_values[colid * n_res_rows + rowid] += value;
            }
        }

        std::vector< std::map<index_type, value_type > > result;
        boost::mpi::all_gather(
                utility::get_communicator(A), col_values, result);

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
};
#endif

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_MIXED_HPP
