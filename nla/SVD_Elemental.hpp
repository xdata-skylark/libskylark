#ifndef SVD_ELEMENTAL_HPP
#define SVD_ELEMENTAL_HPP

#include <elemental.hpp>
#include <skylark.hpp>

namespace skylark {
namespace nla {

#if 0

/** General template */
/**
 * These are the parameters that are needed for sketching SVD.
 * (1) The sketching type to use.
 * (2) The number of singular vectors of the sketched matrix to use.
 *
 * Use:
 * [U_k, S_k, V_k] ~= sketched_SVD (A, k) 
 */ 
template <typename InputMatrixType,
          typename OutputMatrixType,  
          /** if we are using for frobenius norm, don't power iterate */
          >
struct sketched_SVD_t {};

/** Specialization */
template <typename ValueType,
          elem::Distribution CD,
          elem::Distribution RD,
          /** if we are using for frobenius norm, don't power iterate */
          >
struct sketched_SVD_t <elem::DistMatrix<ValueType, CD, RD>,
                       elem::DistMatrix<ValueType, CD, RD> > {
  typedef elem::DistMatrix<ValueType, CD, RD> mpi_matrix_t;
  
  /**
   * \brief Give a brief description here.
   * \param[in] k
   * \param[in] s 
   * \param[in] p  lkjslajslsad
   * \param[in] A
   * \param[out] U
   * \param[out] S
   * \param[out] V
   */
  static void apply (int k,
                     int s,
                     int p,
                     const mpi_matrix_t& A, 
                     mpi_matrix_t& U,
                     mpi_matrix_t& S,
                     mpi_matrix_t& V) {
    /** First check that K < min(M,N), where A is (M,N) */

    /** Next check that S <= M and S>= K, where A is (M,N) */

    /** Check that 'p' is reasonable */

    /** Execute the algorithm */

  }
};


// TODO this should be templated ASAP. They confuse codes that want to define these later.
typedef elem::Matrix<double> MatrixType;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> DistMatrixType;

// Takes an m x nA matrix A, m x nB matrix B and computes C = A'*B which is
// small nA x nB.  A and B are row-partitioned together. So computation of C
// boils down to C = sum_i A_i^T*B_i i.e. an mpi reduce operation.  Note: we
// need to pass the context so we can call mpi::reduce with the associated
// communicator.
void Gemm(DistMatrixType& A, DistMatrixType& B, MatrixType& C, skylark::sketch::context_t& context) {

    int mA = A.Height();
    int nA = A.Width();
    int mB = B.Height();
    int nB = B.Width();

    MatrixType C_local(nA, nB);
    Gemm(elem::ADJOINT, elem::NORMAL, 1.0, A.LockedMatrix(), B.LockedMatrix(), 0.0, C_local);
    boost::mpi::reduce (context.comm,
        C_local.LockedBuffer(),
        C_local.MemorySize(),
        C.Buffer(),
        std::plus<double>(),
        0);
}

// Takes a m x n distributed matrix A and a n x l local matrix B and computes the distributed matrix A*B by local matrix multiplication.
void Gemm(DistMatrixType& A, MatrixType& B, DistMatrixType& C) {
    Gemm(elem::NORMAL, elem::NORMAL, 1.0, A.LockedMatrix(), B, 0.0, C.Matrix());
}

//templatize later
void SVD(DistMatrixType& A, DistMatrixType& U, MatrixType& s,  MatrixType& V, int l, int q, skylark::sketch::context_t& context) {

    int m = A.Height();
    int n = A.Width();

    // Create an n x l JLT Sketch
    skylark::sketch::JLT_t<DistMatrixType, DistMatrixType> JLT (n, l, context);

    // Create space to hold the sketched result
    DistMatrixType Y(m,l);
    //Y.ResizeTo(m,l);

    JLT.apply (A, Y, skylark::sketch::rowwise_tag());

    // TO DO : need to do power iterations here

    // call Explicit QR on Y. Y is overwritten with Q where Y = QR.
    // NOTE: Type conversions below.
    elem::DistMatrix<double> Q(Y);
    elem::QR( Q );
    DistMatrixType Q2(Q);
    Q2 = Q;

    //Compute B = Q'A of size l x n

    MatrixType B(l, n);
    Gemm(Q2, A, B, context);

    // Get SVD of B - Note B is overwritten by U where B = U diag(s) V' is the SVD of B.
    elem::SVD(B, s, V);

    // Write U = Q B
    Gemm(Q2, B, U);
}
#endif

} //nla namespace
} //skylark namespace

#endif // SVD_HPP

