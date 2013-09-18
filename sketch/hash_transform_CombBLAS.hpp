#ifndef SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP
#define SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP

#include <CombBLAS.h>
#include "../utility/external/FullyDistMultiVec.hpp"
#include "../utility/exception.hpp"

#include "context.hpp"
#include "transforms.hpp"
#include "hash_transform_data.hpp"

namespace skylark { namespace sketch {

template <typename IndexType,
          typename ValueType,
          typename IdxDistributionType,
          template <typename> class ValueDistributionType>
struct hash_transform_t <FullyDistMultiVec<IndexType, ValueType>,
                         FullyDistMultiVec<IndexType, ValueType>,
                         IdxDistributionType,
                         ValueDistributionType > :
  public hash_transform_data_t<IndexType,
                               ValueType,
                               IdxDistributionType,
                               ValueDistributionType> {
  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef FullyDistVec<IndexType, ValueType> mpi_vector_t;
  typedef FullyDistMultiVec<IndexType, ValueType> mpi_multi_vector_t;
  typedef hash_transform_data_t<IndexType,
                                ValueType,
                                IdxDistributionType,
                                ValueDistributionType> base_data_t;

  hash_transform_t (int N, int S, skylark::sketch::context_t& context) :
        base_data_t (N, S, context) {}

  template <typename InputMatrixType,
            typename OutputMatrixType>
  hash_transform_t (hash_transform_t<InputMatrixType,
                                     OutputMatrixType,
                                     IdxDistributionType,
                                     ValueDistributionType>& other) :
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

  void apply_impl_single (mpi_vector_t& a,
                          mpi_vector_t& sketch_of_a,
                          columnwise_tag) {
    std::vector<value_t> sketch_term(base_data_t::S,0);

    /** Accumulate the local sketch vector */
    /** FIXME: Lot's of random access --- not good for performance */
    DenseVectorLocalIterator<index_t, value_t> local_iter(a);
    while(local_iter.HasNext()) {
      index_t idx = local_iter.GetLocIndex();
      index_t global_idx = local_iter.LocalToGlobal(idx);
      index_t global_sketch_idx = base_data_t::row_idx[global_idx];
      sketch_term[global_sketch_idx] +=
            (local_iter.GetValue()*base_data_t::row_value[global_idx]);
      local_iter.Next();
    }

    /** Accumulate the global sketch vector */
    /** FIXME: Only need to scatter ... don't need everything everywhere */
    MPI_Allreduce(MPI_IN_PLACE,
                  &(sketch_term[0]),
                  base_data_t::S,
                  MPIType<value_t>(),
                  MPI_SUM,
                  a.commGrid->GetWorld());

    /** Fill in .. SetElement() is dummy for non-local sets, so it's ok */
    for (index_t i=0; i<base_data_t::S; ++i) {
      sketch_of_a.SetElement(i,sketch_term[i]);
    }
  }

  void apply_impl (mpi_multi_vector_t& A,
                   mpi_multi_vector_t& sketch_of_A,
                   columnwise_tag) {
    const index_t num_rhs = A.size;
    if (sketch_of_A.size != num_rhs) { /** error */; return; }
    if (A.dim != base_data_t::N) { /** error */; return; }
    if (sketch_of_A.dim != base_data_t::S) { /** error */; return; }

    /** FIXME: Can sketch all the vectors in one shot */
    for (index_t i=0; i<num_rhs; ++i) {
      apply_impl_single (A[i], sketch_of_A[i], columnwise_tag());
    }
  }
};

template <typename IndexType,
      typename ValueType,
      typename IdxDistributionType,
      template <typename> class ValueDistributionType>
struct hash_transform_t <
      SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
      SpParMat<IndexType, ValueType, SpDCCols<IndexType, ValueType> >,
      IdxDistributionType,
      ValueDistributionType > :
  public hash_transform_data_t<IndexType,
                               ValueType,
                               IdxDistributionType,
                               ValueDistributionType> {
  typedef IndexType index_t;
  typedef ValueType value_t;
  typedef SpDCCols< IndexType, value_t > col_t;
  typedef FullyDistVec< IndexType, ValueType > mpi_vector_t;
  typedef SpParMat< IndexType, value_t, col_t > matrix_t;
  typedef SpParMat< IndexType, value_t, col_t > output_matrix_t;
  typedef hash_transform_data_t<IndexType,
                                ValueType,
                                IdxDistributionType,
                                ValueDistributionType> base_data_t;

  hash_transform_t (int N, int S, skylark::sketch::context_t& context) :
                  base_data_t(N, S, context) {}

  template <typename InputMatrixType,
            typename OutputMatrixType>
  hash_transform_t (hash_transform_t<InputMatrixType,
                                     OutputMatrixType,
                                     IdxDistributionType,
                                     ValueDistributionType>& other) :
              base_data_t(other.get_data()) {}

  template <typename Dimension>
  void apply (matrix_t &A,
              output_matrix_t &sketch_of_A,
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

  /**
   * Apply the sketching transform that is described in by the sketch_of_A.
   * Implementation for the column-wise direction of sketching.
   *
   * FIXME: This is really inefficient. So, we need something better.
   * The code duplication can also be eliminated here.
   */
  void apply_impl (matrix_t &A,
                   output_matrix_t &sketch_of_A,
                   skylark::sketch::columnwise_tag) {

    const size_t rank = A.getcommgrid()->GetRank();

    // extract columns of matrix
    col_t &data = A.seq();

    //FIXME: next step only store local generated non-zeros
    const size_t ncols = sketch_of_A.getncol();
    const size_t nrows = sketch_of_A.getnrow();
    const size_t matrix_size = ncols * nrows;
    mpi_vector_t cols(matrix_size);
    mpi_vector_t rows(matrix_size);
    mpi_vector_t vals(matrix_size);
    std::vector<value_t> my_vals(matrix_size, 0.0);

    for(index_t i = 0; i < matrix_size; ++i) {
        rows.SetElement(i, static_cast<index_t>(i / ncols));
        cols.SetElement(i, i % ncols);
    }

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

        const index_t rowid = nz.rowid();
        const index_t colid = col.colid();
        index_t pos = (colid + my_col_offset) + 1.0 * ncols *
                      base_data_t::row_idx[rowid + my_row_offset];

        my_vals[pos] += nz.value() *
                        base_data_t::row_value[rowid + my_row_offset];
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &(my_vals[0]), static_cast<int>(matrix_size),
                  boost::mpi::get_mpi_datatype<value_t>(), MPI_SUM,
                  A.getcommgrid()->GetWorld());

    for(size_t i = 0; i < matrix_size; i++)
        vals.SetElement(i, my_vals[i]);

    output_matrix_t tmp(sketch_of_A.getnrow(),
                        sketch_of_A.getncol(),
                        rows,
                        cols,
                        vals);

    sketch_of_A = tmp;
  }

  /**
   * Apply the sketching transform that is described in by the sketch_of_A.
   * Implementation for the row-wise direction of sketching.
   */
  void apply_impl (matrix_t &A,
                   output_matrix_t &sketch_of_A,
                   skylark::sketch::rowwise_tag) {

    const size_t rank = A.getcommgrid()->GetRank();

    // extract columns of matrix
    col_t &data = A.seq();

    //FIXME: next step only store local generated non-zeros
    const size_t ncols = sketch_of_A.getncol();
    const size_t nrows = sketch_of_A.getnrow();
    const size_t matrix_size = ncols * nrows;
    mpi_vector_t cols(matrix_size);
    mpi_vector_t rows(matrix_size);
    mpi_vector_t vals(matrix_size);
    std::vector<value_t> my_vals(matrix_size, 0.0);

    for(index_t i = 0; i < matrix_size; ++i) {
        rows.SetElement(i, static_cast<index_t>(i / ncols));
        cols.SetElement(i, i % ncols);
    }

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

        const index_t rowid = nz.rowid();
        const index_t colid = col.colid();
        index_t pos = (rowid + my_row_offset) * ncols +
                      base_data_t::row_idx[colid + my_col_offset];

        my_vals[pos] += nz.value() *
                        base_data_t::row_value[colid + my_col_offset];

      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &(my_vals[0]), static_cast<int>(matrix_size),
                  boost::mpi::get_mpi_datatype<value_t>(), MPI_SUM,
                  A.getcommgrid()->GetWorld());

    for(size_t i = 0; i < matrix_size; i++)
        vals.SetElement(i, my_vals[i]);

    output_matrix_t tmp(sketch_of_A.getnrow(),
                        sketch_of_A.getncol(),
                        rows,
                        cols,
                        vals);

    sketch_of_A = tmp;
  }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HASH_TRANSFORM_COMBBLAS_HPP
