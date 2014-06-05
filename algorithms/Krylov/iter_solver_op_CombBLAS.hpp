#ifndef SKYLARK_ITER_SOLVER_OP_COMBBLAS_HPP
#define SKYLARK_ITER_SOLVER_OP_COMBBLAS_HPP

/*****
 *****    NOTE: THIS CLASS IS NOT USED ANYMORE!
 *****    It is kept until the useful code can be transfered
 *****    as functions in namespace base.
 *****/

#include <functional>
#include <algorithm>
#include <CombBLAS.h>

namespace skylark { namespace nla {

template <typename IndexType, 
          typename ValueType>
struct iter_solver_op_t <SpParMat<IndexType, 
                                  ValueType, 
                                  SpDCCols<IndexType, ValueType> >,
                         FullyDistMultiVec<IndexType, ValueType> > {
  /** A class that allows us to compute power of values */
  template <int p, bool invert>
  struct pow_t {
    typedef ValueType argument_type;
    typedef ValueType result_type;

    static result_type apply(const argument_type& arg) { 
      return ((false==invert) ? pow(arg,p):pow(arg,1./p)); 
    }

    result_type operator()(const argument_type& arg) const { 
      return apply(arg); 
    }
  };

  struct scale_t {
    typedef ValueType argument_type;
    typedef ValueType result_type;
    
    const argument_type scaling_factor;

    scale_t (argument_type scaling_factor): scaling_factor(scaling_factor) {}

    result_type operator()(const argument_type& arg) const { 
      return scaling_factor*arg; 
    }
  };

  template <typename BinaryOperation>
  struct scaled_bin_op_t {
    typedef ValueType argument_type;
    typedef ValueType result_type;
    
    BinaryOperation bin_op;
    argument_type scale_arg_1;
    argument_type scale_arg_2;

    scaled_bin_op_t (argument_type scale_arg_1,
                     argument_type scale_arg_2) : scale_arg_1(scale_arg_1),
                                                  scale_arg_2(scale_arg_2) {}

    result_type operator()(const argument_type& arg_1,
                           const argument_type& arg_2) {
      return bin_op(scale_arg_1*arg_1, scale_arg_2*arg_2);
    }
  };

  /** Typedefs for some variables */
  typedef IndexType index_t; 
  typedef ValueType value_t;
  typedef SpDCCols<IndexType, ValueType> seq_matrix_t;
  typedef FullyDistMultiVec<IndexType, ValueType> mpi_multi_vector_t;
  typedef typename mpi_multi_vector_t::mpi_vector_t mpi_vector_t;
  typedef SpParMat<IndexType, 
                   ValueType, 
                   SpDCCols<IndexType, ValueType> > mpi_matrix_t;
  typedef PlusTimesSRing<ValueType,ValueType> semi_ring_t;
  typedef pow_t<2, false> square_t;
  typedef pow_t<2, true> square_root_t;
  typedef scaled_bin_op_t<std::plus<value_t> > ax_plus_by_t;

  /************************************************************************/
  /*                     TYPEDEFs -- if needed externally                 */
  /************************************************************************/
  typedef index_t index_type;
  typedef value_t value_type;
  typedef mpi_matrix_t matrix_type;
  typedef mpi_vector_t vector_type;
  typedef mpi_multi_vector_t multi_vector_type;

  /*************************************************************************/
  /*          Operations between a matrix and a multi-vector               */
  /*                            UNOPTIMIZED                                */
  /*************************************************************************/
  static mpi_matrix_t transpose (const mpi_matrix_t& A) { 
    mpi_matrix_t AT = A;
    AT.Transpose();
    return AT;
  }

  static void mat_vec (const mpi_matrix_t& A, 
                       const mpi_multi_vector_t& X,
                       mpi_multi_vector_t& AX) {
    /** Get the dimensions and make sure that they agree */
    index_t m = A.getnrow();
    index_t n = A.getncol();

    if (n != X.dim) { /* error */ }

    /** As SpMM is not provided, do the multiplication one at a time */
    for (index_t i=0; i<X.size; ++i) AX[i] = SpMV<semi_ring_t>(A,X[i]);
  }

  template <typename RandomAccessContainer>
  static void ax_plus_by (const RandomAccessContainer& a,
                          mpi_multi_vector_t& X,
                          const RandomAccessContainer& b,
                          const mpi_multi_vector_t& Y) {
    /** Basic error checking on the dimensions for the multi-vectors */
    if (X.dim != Y.dim) { /* error */ }
    if (X.size != Y.size) { /* error */ }
    if (a.size() != X.size) { /* error */ }
    if (b.size() != Y.size) { /* error */ }

    /** Perform all the required element-wise operations one by one */
    for (index_t i=0; i<X.size; ++i) 
      X[i].EWiseApply(Y[i], ax_plus_by_t(a[i], b[i]));
  }

  template <typename RandomAccessContainer>
  static void scale (const RandomAccessContainer& a,
                     mpi_multi_vector_t& X) {
    if (a.size() != X.size) { /* error */ }
    for (index_t i=0; i<X.size; ++i) X[i].Apply(scale_t(a[i]));
  }

  template <typename RandomAccessContainer>
  static void norm (const mpi_multi_vector_t& X,
                    RandomAccessContainer& norms) {
    for (index_t i=0; i<X.size; ++i) norms[i] = square_root_t::apply 
      (X[i].Reduce(std::plus<double>(), 0.0, square_t()));
  }

  static void get_dim (const mpi_multi_vector_t& X, index_t& M, index_t& N) {
    M = X.dim; 
    N = X.size;
  }

  static void get_dim (const mpi_matrix_t& A, index_t& M, index_t& N) {
    M = A.getnrow();
    N = A.getncol();
  }

  template <typename RandomAccessContainer>
  static void residual_norms (const mpi_matrix_t& A,
                              const mpi_multi_vector_t& B,
                              const mpi_multi_vector_t& X,
                              RandomAccessContainer& r_norms) {
    mpi_multi_vector_t R(B);
    
    /** 1. Compute AX so that we can subtract B from it */
    mat_vec (A, X, R);

    /** 2. Subtract AX from B to form R */
    RandomAccessContainer ONES(X.size, 1.0);
    RandomAccessContainer MINUS_ONES(X.size, -1.0);
    ax_plus_by (MINUS_ONES, R, ONES, B);

    /** 3. Compute the norms of the residual */
    norm (R, r_norms);
  }
};

} } /** namespace skylark::nla */

#endif  // SKYLARK_ITER_SOLVER_OP_COMBBLAS_HPP
