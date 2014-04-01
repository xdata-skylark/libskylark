#ifndef SKYLARK_SKETCHED_SVD_ELEMENTAL_HPP
#define SKYLARK_SKETCHED_SVD_ELEMENTAL_HPP

#include <elemental.hpp>
#include "../base/context.hpp"
#include "../utility/exception.hpp"
#include "../utility/external/get_communicator.hpp"

namespace skylark { namespace nla {
#if 0
/** Specialization */
template <typename ValueType>
struct sketch_svd_t <elem::DistMatrix<ValueType, elem::MC, elem::MR>,
                     elem::DistMatrix<ValueType, elem::MC, elem::MR>,
                     skylark::sketch::JLT_t <
                         elem::DistMatrix<ValueType, elem::MC, elem::MR>,
                         elem::DistMatrix<ValueType, elem::MC, elem::MR> > > {
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::MC, elem::MR> matrix_type;
    typedef elem::DistMatrix<value_type, elem::MC, elem::MR> output_matrix_type;
    typedef SketchTransformType sketch_transform_type;
    typedef sketch_transform_type::output_matrix_type sketched_matrix_type;

    /**
     * \param[in]  k           target rank of the approximate decomposition
     * \param[in]  sketch_size number of columns of the sketched matrix
     * \param[in]  q           number of subspace iterations
     * \param[in]  A           input matrix
     * \param[out] U           approximate left singular vectors output matrix
     * \param[out] S           approximate singular values output matrix
     * \param[out] V           approximate right singular vectors output matrix
     *
     * FIXME: Provide for the case of column-wise sketching?
     */
    static void apply (int k,
                       int sketch_size,
                       int q,
                       const matrix_type& A,
                       output_matrix_type& U,
                       output_matrix_type& S,
                       output_matrix_type& V,
                       skylark::base::context_t& context) {

        int height = A.Height();
        int width = A.Width();

        /**
         * Sanity checks, raise an exception if:
         *   i)   the target rank is too large for the given input matrix or
         *   ii)  the number of columns of the sketched matrix either:
         *        - exceeds its width or
         *        - is less than the target rank
         */
         if (k > std::min(height, width)) ||
            (sketch_size > width) ||
                (sketch_size < k) {
                SKYLARK_THROW_EXCEPTION(
                    utility::elemental_exception()
                        << utility::error_msg(e.what()) );
          }

         /** Apply the sketching transformation rowwise:
          *  A (height x width) -> Y (height x sketch_size)
         */
         sketch_transform_type sketch_transform(width, sketch_size, context);
         sketched_matrix_type Y(height, sketch_size);
         sketch_transform.apply(A, Y, skylark:sketch::rowwise_tag());

         /** Q, _ = numpy.linalg.qr(Y); (Python) */
         sketched_matrix_type Q(Y);
         elem::qr::Explicit(Q);

         /** q steps of subspace iteration */
         for(int step = 0; step < q; step++) {
             /** Y = numpy.dot(A.T, Q); Q, _ = numpy.linalg.qr(Y); (Python) */
             elem::Gemm(elem::ADJOINT, elem::NORMAL, 0.0, A, Q, Y);
             sketched_matrix_type Q(Y);
             elem::qr::Explicit(Q);

             /** Y = numpy.dot(A, Q); Q, _ = numpy.linalg.qr(Y);   (Python) */
             elem::Gemm(elem::NORMAL, elem::NORMAL, 0.0, A, Q, Y);
             sketched_matrix_type Q(Y);
             elem::qr::Explicit(Q);
         }

         /** B = numpy.dot(Q.T, A); (Python) */
         sketched_matrix_type B;
         elem::Gemm(elem::ADJOINT, elem::NORMAL, value_type(0), Q, A, B);

         /** U, sigma, Vt = numpy.linalg.svd(B); V = Vt.T; (Python) */
         elem::DistMatrix<elem::VR, elem::STAR> Sigma;
         elem::SVD(B, Sigma, V);
         S = Sigma;

         /** U = B; U = numpy.dot(Q, U); (Python) */
         elem::Gemm(elem::NORMAL, elem::NORMAL, Q, B, value_type(0), U);
  }
};
#endif

#if 0
/*************************************************************************/
/* Everything below here is Vikas' code; I am keeping it for posterity   */
/*************************************************************************/

// TODO this should be templated ASAP. They confuse codes that want to define
// these later.
typedef elem::Matrix<double> MatrixType;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> DistMatrixType;

// Takes an m x nA matrix A, m x nB matrix B and computes C = A'*B which is
// small nA x nB.  A and B are row-partitioned together. So computation of C
// boils down to C = sum_i A_i^T*B_i i.e. an mpi reduce operation.  Note: we
// need to pass the context so we can call mpi::reduce with the associated
// communicator.
void Gemm(DistMatrixType& A,
          DistMatrixType& B,
          MatrixType& C,
          skylark::base::context_t& context) {

    int mA = A.Height();
    int nA = A.Width();
    int mB = B.Height();
    int nB = B.Width();

    MatrixType C_local(nA, nB);
    Gemm(elem::ADJOINT,
         elem::NORMAL,
         1.0,
         A.LockedMatrix(),
         B.LockedMatrix(),
         0.0,
         C_local);

    // get communicator from matrix
    boost::mpi::communicator comm = skylark::utility::get_communicator(A);

    boost::mpi::reduce (comm,
                        C_local.LockedBuffer(),
                        C_local.MemorySize(),
                        C.Buffer(),
                        std::plus<double>(),
                        0);
}

// Takes a m x n distributed matrix A and a n x l local matrix B and computes the distributed matrix A*B by local matrix multiplication.
void Gemm(DistMatrixType& A, MatrixType& B, DistMatrixType& C) {
    Gemm(elem::NORMAL,
         elem::NORMAL,
         1.0,
         A.LockedMatrix(),
         B,
         0.0,
         C.Matrix());
}

//templatize later
void SVD(DistMatrixType& A,
         DistMatrixType& U,
         MatrixType& s,
         MatrixType& V,
         int l,
         int q,
         skylark::base::context_t& context) {

    int m = A.Height();
    int n = A.Width();

    // Create an n x l JLT Sketch
    skylark::sketch::JLT_t<DistMatrixType, DistMatrixType> JLT (n, l, context);

    // Create space to hold the sketched result
    DistMatrixType Y(m,l);
    //Y.Resize(m,l);

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

    // Get SVD of B - Note B is overwritten by U where B = U diag(s) V' is the
    // SVD of B.
    elem::SVD(B, s, V);

    // Write U = Q B
    Gemm(Q2, B, U);
}
#endif

} } /** namespace skylark::nla */

#endif // SKYLARK_SKETCHED_SVD_ELEMENTAL_HPP
