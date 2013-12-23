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

    template <typename Dimension>
    void apply (mpi_multi_vector_t &A,
        mpi_multi_vector_t &sketch_of_A,
        Dimension dimension) {
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
    void apply_impl_single (mpi_vector_t& a,
        mpi_vector_t& sketch_of_a,
        columnwise_tag) {
        std::vector<value_type> sketch_term(base_data_t::S,0);

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


    void apply_impl (mpi_multi_vector_t& A,
        mpi_multi_vector_t& sketch_of_A,
        columnwise_tag) {
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

        // build local mapping (global_idx, value) first
        //FIXME: come up with more efficient data structure
        typedef std::map<size_t, value_type> sp_mat_value_t;
        std::map<size_t, value_type> my_vals_map;

        const size_t my_row_offset =
            static_cast<int>(0.5 + (static_cast<double>(A.getnrow()) /
                    A.getcommgrid()->GetGridRows())) *
            A.getcommgrid()->GetRankInProcCol(rank);

        const size_t my_col_offset =
            static_cast<int>(0.5 + (static_cast<double>(A.getncol()) /
                    A.getcommgrid()->GetGridCols())) *
            A.getcommgrid()->GetRankInProcRow(rank);

        for(typename col_t::SpColIter col = data.begcol();
            col != data.endcol(); col++) {
            for(typename col_t::SpColIter::NzIter nz = data.begnz(col);
                nz != data.endnz(col); nz++) {

                const index_type rowid = nz.rowid()  + my_row_offset;
                const index_type colid = col.colid() + my_col_offset;

                index_type pos   = getPos(rowid, colid, ncols, dist);
                value_type value = nz.value() * getRowValue(rowid, colid, dist);

                if(my_vals_map.count(pos) != 0)
                    my_vals_map[pos] += value;
                else
                    my_vals_map.insert(std::pair<size_t,
                                                 value_type>(pos, value));
            }
        }

        // aggregate values
        boost::mpi::communicator world(A.getcommgrid()->GetWorld(),
            boost::mpi::comm_duplicate);
        std::vector< std::map<size_t, value_type> > vector_of_maps;

        //FIXME: best to selectively send to exchange pair of (size, [double])
        //       with processor that needs the values. It should be possible to
        //       pre-compute the ranges of positions that are kept
        //       on a processor.
        boost::mpi::all_gather< std::map<size_t, value_type> >(world,
            my_vals_map,
            vector_of_maps );

        // re-sort/insert in value map
        std::map<size_t, value_type> vals_map;
        typename std::map<size_t, value_type>::iterator itr;
        for(size_t i = 0; i < vector_of_maps.size(); ++i) {

            for(itr = vector_of_maps[i].begin(); itr != vector_of_maps[i].end();
                itr++) {
                if(vals_map.count(itr->first) != 0)
                    vals_map[itr->first] += itr->second;
                else
                    vals_map.insert(std::pair<size_t,
                                              value_type>(itr->first,
                                                  itr->second));
            }
        }

        // .. and finally create a new sparse matrix
        const size_t matrix_size = vals_map.size();
        mpi_vector_t cols(matrix_size);
        mpi_vector_t rows(matrix_size);
        mpi_vector_t vals(matrix_size);
        size_t idx = 0;

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
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP
