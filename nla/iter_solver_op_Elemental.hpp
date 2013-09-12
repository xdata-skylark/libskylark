#ifndef SKYLARK_ITER_SOLVER_OP_ELEMENTAL_HPP
#define SKYLARK_ITER_SOLVER_OP_ELEMENTAL_HPP

#include <functional>
#include <algorithm>
#include <elemental.hpp>
#include <mpi.h>

namespace skylark { namespace nla {

/***************************************************************************/
/*            SOLVER FOR elem::DistMatrix; needs Gemm capability           */
/***************************************************************************/
template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RD>
struct iter_solver_op_t <elem::DistMatrix<ValueType, CD, RD>,
                         elem::DistMatrix<ValueType, CD, RD> > {

  /** Typedefs for some variables */
  typedef int index_t; 
  typedef ValueType value_t;
  typedef elem::DistMatrix<ValueType, CD, RD> mpi_multi_vector_t;
  typedef mpi_multi_vector_t mpi_matrix_t;
  typedef elem::Matrix<ValueType> matrix_t; /**< used internally */

  /************************************************************************/
  /*                     TYPEDEFs -- if needed externally                 */
  /************************************************************************/
  typedef index_t index_type;
  typedef value_t value_type;
  typedef mpi_matrix_t matrix_type;
  typedef mpi_multi_vector_t multi_vector_type;

  /*************************************************************************/
  /*          Operations between a matrix and a multi-vector               */
  /*                            UNOPTIMIZED                                */
  /*************************************************************************/
  static mpi_matrix_t transpose (const mpi_matrix_t& A) { 
    /** Stupid way to do it --- think of something better */
    mpi_matrix_t AT(A.Grid());
    elem::Transpose(A, AT);
    return AT;
  }

  static void mat_vec (const mpi_matrix_t& A, 
                       const mpi_multi_vector_t& X,
                       mpi_multi_vector_t& AX) {
    /** Get the dimensions and make sure that they agree */
    index_t m = A.Height();
    index_t n = A.Width();

    if (n != X.Height()) { /* error */ }

    /** As GEMM is provided, do the multiplication together */
    elem::Gemm(elem::NORMAL, elem::NORMAL, 1.0, A, X, 0.0, AX);
  }

  static void make_dist_from_local (const matrix_t& local,
                                    mpi_matrix_t& global) {
    for (index_t jLocal=0; jLocal<global.LocalWidth(); ++jLocal) {
      const index_t j = global.RowShift() + jLocal*global.RowStride();
      for(index_t iLocal=0; iLocal<global.LocalHeight(); ++iLocal) {
        const index_t i=global.ColShift() + iLocal*global.ColStride();
        global.SetLocal(iLocal, jLocal, local.Get(i,j));
      }
    }
  }

  template <typename RandomAccessContainer>
  static void scale (const RandomAccessContainer& a,
                     mpi_multi_vector_t& X) {
    if (a.size() != X.Width()) { /* error */ }

    /** 
     * We need each distributed column to be multiplied by corresponding 
     * entry in the vector 'a'. So, we use Elemental's diagonal scale 
     * operation.
     */ 
    matrix_t local_a (X.Width(), 1, &(a[0]), X.Width());
    mpi_multi_vector_t DA (X.Width(), 1, X.Grid());
    make_dist_from_local (local_a, DA);
    elem::DiagonalScale (elem::RIGHT, elem::NORMAL, DA, X);
  }

  template <typename RandomAccessContainer>
  static void ax_plus_by (const RandomAccessContainer& a,
                          mpi_multi_vector_t& X,
                          const RandomAccessContainer& b,
                          const mpi_multi_vector_t& Y) {
    /** Basic error checking on the dimensions for the multi-vectors */
    if (X.Width() != Y.Width()) { /* error */ }
    if (X.Height() != Y.Height()) { /* error */ }
    if (a.size() != X.Width()) { /* error */ }
    if (b.size() != Y.Width()) { /* error */ }

    /** 
     * We solve y = ax+by in three steps:
     * (1) y = ay --- this can be done using Elemental's diagonal scale.
     * (2) x = bx --- this  can be done using Elemental's diagonal scale.
     * (3) y = x+y --- this is done using Elemental's axpy BLAS call.
     */ 
    scale (a, X);
    mpi_multi_vector_t copy_of_Y (Y); scale (b, copy_of_Y);
    elem::Axpy (1.0, copy_of_Y, X);
  }

  template <typename RandomAccessContainer>
  static void norm (const mpi_multi_vector_t& X,
                    RandomAccessContainer& norms) {
    /** 
     * There is no column norm operation in Elemental, so for now, we are
     * doing this by hand. Basically, everyone accumulates the column entries
     * they own, then do an all reduce to get the global sum --- pretty simple.
     */ 
    for (index_t i=0; i<norms.size(); ++i) norms[i] = 0.0;
    for (index_t jLocal=0; jLocal<X.LocalWidth(); ++jLocal) {
      const index_t j = X.RowShift() + jLocal*X.RowStride();
      value_t local_column_sum = 0.0;
      for(index_t iLocal=0; iLocal<X.LocalHeight(); ++iLocal) {
        local_column_sum += pow(X.GetLocal(iLocal, jLocal), 2);
      }
      norms[j] = local_column_sum;
    }

#if 1
    /** This is usgly still */
    MPI_Allreduce (MPI_IN_PLACE, 
                   &(norms[0]), 
                   norms.size(), 
                   boost::mpi::get_mpi_datatype<value_t>(),
                   MPI_SUM,
                   MPI_Comm (X.Grid().Comm()));
#else
    /* Fix this instead */
    norms = boost::mpi::all_reduce (MPI_Comm(X.Grid().Comm()), 
                                    norms, 
                                    std::plus<value_type>());

#endif
     
    for (index_t i=0; i<norms.size(); ++i) norms[i] = sqrt(norms[i]);
  }

  static void get_dim (const mpi_multi_vector_t& X, index_t& M, index_t& N) {
    M = X.Height(); 
    N = X.Width();
  }

  template <typename RandomAccessContainer>
  static void residual_norms (const mpi_matrix_t& A,
                              const mpi_multi_vector_t& B,
                              const mpi_multi_vector_t& X,
                              RandomAccessContainer& r_norms) {
    mpi_multi_vector_t R(B);
    
    /** 1. Compute AX-B in one shot */
    elem::Gemm (elem::NORMAL, elem::NORMAL, 1.0, A, X, -1.0, R);

    /** 2. Compute the norms of the residual */
    norm (R, r_norms);
  }
};

/***************************************************************************/
/*            SOLVER FOR elem::Matrix --- SINGLE NODE                      */
/***************************************************************************/
template <typename ValueType>
struct iter_solver_op_t <elem::Matrix<ValueType>, elem::Matrix<ValueType> > {

  /** Typedefs for some variables */
  typedef int index_t; 
  typedef ValueType value_t;
  typedef elem::Matrix<ValueType> multi_vector_t;
  typedef elem::Matrix<ValueType> matrix_t;

  /************************************************************************/
  /*                     TYPEDEFs -- if needed externally                 */
  /************************************************************************/
  typedef index_t index_type;
  typedef value_t value_type;
  typedef matrix_t matrix_type;
  typedef multi_vector_t multi_vector_type;

  /*************************************************************************/
  /*          Operations between a matrix and a multi-vector               */
  /*                            UNOPTIMIZED                                */
  /*************************************************************************/
  static matrix_t transpose (const matrix_t& A) { 
    /** Stupid way to do it --- think of something better */
    matrix_t AT;
    elem::Transpose(A, AT);
    return AT;
  }

  static void mat_vec (const matrix_t& A, 
                       const multi_vector_t& X,
                       multi_vector_t& AX) {
    /** Get the dimensions and make sure that they agree */
    index_t m = A.Height();
    index_t n = A.Width();

    if (n != X.Height()) { /* error */ }

    /** As GEMM is provided, do the multiplication together */
    elem::Gemm(elem::NORMAL, elem::NORMAL, 1.0, A, X, 0.0, AX);
  }

  template <typename RandomAccessContainer>
  static void scale (const RandomAccessContainer& a,
                     multi_vector_t& X) {
    if (a.size() != X.Width()) { /* error */ }

    /** 
     * We need each distributed column to be multiplied by corresponding 
     * entry in the vector 'a'. So, we use Elemental's diagonal scale 
     * operation.
     */ 
    matrix_t local_a (X.Width(), 1, &(a[0]), X.Width());
    elem::DiagonalScale (elem::RIGHT, elem::NORMAL, local_a, X);
  }

  template <typename RandomAccessContainer>
  static void ax_plus_by (const RandomAccessContainer& a,
                          multi_vector_t& X,
                          const RandomAccessContainer& b,
                          const multi_vector_t& Y) {
    /** Basic error checking on the dimensions for the multi-vectors */
    if (X.Width() != Y.Width()) { /* error */ }
    if (X.Height() != Y.Height()) { /* error */ }
    if (a.size() != X.Width()) { /* error */ }
    if (b.size() != Y.Width()) { /* error */ }

    /** 
     * We solve y = ax+by in three steps:
     * (1) y = ay --- this can be done using Elemental's diagonal scale.
     * (2) x = bx --- this  can be done using Elemental's diagonal scale.
     * (3) y = x+y --- this is done using Elemental's axpy BLAS call.
     */ 
    scale (a, X);
    multi_vector_t copy_of_Y (Y); scale (b, copy_of_Y);
    elem::Axpy (1.0, copy_of_Y, X);
  }

  template <typename RandomAccessContainer>
  static void norm (const multi_vector_t& X,
                    RandomAccessContainer& norms) {
    /** 
     * There is no column norm operation in Elemental, so for now, we are
     * doing this by hand. Basically, everyone accumulates the column entries
     * they own, then do an all reduce to get the global sum --- pretty simple.
     */ 
    for (index_t i=0; i<norms.size(); ++i) norms[i] = 0.0;
    for (index_t j=0; j<X.Width(); ++j) {
      value_t local_column_sum = 0.0;
      for(index_t i=0; i<X.Height(); ++i) {
        local_column_sum += pow(X.Get(i, j), 2);
      }
      norms[j] = local_column_sum;
    }
     
    for (index_t i=0; i<norms.size(); ++i) norms[i] = sqrt(norms[i]);
  }

  static void get_dim (const multi_vector_t& X, index_t& M, index_t& N) {
    M = X.Height(); 
    N = X.Width();
  }

  template <typename RandomAccessContainer>
  static void residual_norms (const matrix_t& A,
                              const multi_vector_t& B,
                              const multi_vector_t& X,
                              RandomAccessContainer& r_norms) {
    multi_vector_t R(B);
    
    /** 1. Compute AX-B in one shot */
    elem::Gemm (elem::NORMAL, elem::NORMAL, 1.0, A, X, -1.0, R);

    /** 2. Compute the norms of the residual */
    norm (R, r_norms);
  }
};

} } /** namespace skylark::nla */

#endif  // SKYLARK_ITER_SOLVER_OP_ELEMENTAL_HPP
