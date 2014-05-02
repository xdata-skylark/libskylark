#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#include <elemental.hpp>
#include <CombBLAS.h>
#include <SpParMat.h>

#include <skylark.hpp>

#include "../../base/Gemm.hpp"
#include "../../base/Gemm_detail.hpp"

/** Typedef DistMatrix and Matrix */
typedef elem::Matrix<double> MatrixType;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistMatrixVCSType;
typedef elem::DistMatrix<double> DistMatrixType;

typedef SpDCCols< size_t, double> col_t;
typedef SpParMat< size_t, double, col_t > cbDistMatrixType;

static const size_t matrix_size = 50;

static MatrixType nn_expected;
static MatrixType tn_expected;
static MatrixType nt_expected;
static MatrixType tt_expected;

template <typename dist_matrix_t>
void check_matrix(const dist_matrix_t &result, const MatrixType &expected,
                  const std::string error) {

    elem::DistMatrix<double, elem::STAR, elem::STAR> full_result = result;
    for(size_t j = 0; j < full_result.Height(); j++ )
        for(size_t i = 0; i < full_result.Width(); i++ ) {
            if(full_result.GetLocal(j, i) != expected.Get(j, i)) {
                std::cout << result.GetLocal(j, i) << " != "
                          << expected.Get(j, i)
                          << " at index (" << j << ", " << i << ")"
                          << std::endl;
                BOOST_FAIL(error.c_str());
            }
        }
}


int test_main(int argc, char *argv[]) {

    namespace mpi = boost::mpi;

#ifdef SKYLARK_HAVE_OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif

    mpi::environment env (argc, argv);
    mpi::communicator world;

    elem::Initialize (argc, argv);
    MPI_Comm mpi_world(world);
    elem::Grid grid (mpi_world);

    // compute local expected value
    MatrixType localA(matrix_size, matrix_size);
    for( size_t j = 0; j < localA.Height(); j++ ) {
        for( size_t i = 0; i < localA.Width(); i++ ) {
            double value = j * matrix_size + i + 1;
            localA.Set(j, i, value);
        }
    }

    elem::Ones(nn_expected, matrix_size, matrix_size);
    elem::Gemm(elem::NORMAL, elem::NORMAL, -1.0, localA, localA,
                1.5, nn_expected);
    elem::Ones(nt_expected, matrix_size, matrix_size);
    elem::Gemm(elem::NORMAL, elem::TRANSPOSE, -1.0, localA, localA,
                1.5, nt_expected);
    elem::Ones(tn_expected, matrix_size, matrix_size);
    elem::Gemm(elem::TRANSPOSE, elem::NORMAL, -1.0, localA, localA,
                1.5, tn_expected);
    elem::Ones(tt_expected, matrix_size, matrix_size);
    elem::Gemm(elem::TRANSPOSE, elem::TRANSPOSE, -1.0, localA, localA,
                1.5, tt_expected);


    // prepare an Elemental matrix with the test data
    double val = 0.0;
    elem::DistMatrix<double, elem::STAR, elem::STAR>
        A_stst(matrix_size, matrix_size, grid);
    for( size_t j = 0; j < A_stst.LocalHeight(); j++ ) {
        for( size_t i = 0; i < A_stst.LocalWidth(); i++ ) {
            val = (j * A_stst.ColStride() + A_stst.ColShift()) * matrix_size +
                   i * A_stst.RowStride() + A_stst.RowShift() + 1;
            A_stst.SetLocal(j, i, val);
        }
    }

    // and fill a CombBLAS sparse matrix (with the same data)
    FullyDistVec<size_t, double> cols(matrix_size * matrix_size, 0.0);
    FullyDistVec<size_t, double> rows(matrix_size * matrix_size, 0.0);
    FullyDistVec<size_t, double> vals(matrix_size * matrix_size, 0.0);

    for(size_t i = 0; i < matrix_size * matrix_size; ++i) {
        rows.SetElement(i, floor(i / matrix_size));
        cols.SetElement(i, i % matrix_size);
        vals.SetElement(i, static_cast<double>(i+1));
    }

    cbDistMatrixType B(matrix_size, matrix_size, rows, cols, vals);



    //std::vector<double> local_matrix;
    //skylark::base::detail::mixed_gemm_local_part_tt(-1.0, B, A_stst, 0.0,
            //local_matrix);
    //for(size_t idx = 0; idx < local_matrix.size(); idx++)
        //std::cout << local_matrix[idx] << std::endl;


    if(world.rank() == 0)
        std::cout << "Testing CombBLAS^T x Elemental (VX/*) = Elemental (*/*) :";

    elem::DistMatrix<double, elem::STAR, elem::STAR>
        result_stst(matrix_size, matrix_size, grid);
    for( size_t j = 0; j < result_stst.LocalHeight(); j++ )
        for( size_t i = 0; i < result_stst.LocalWidth(); i++ )
            result_stst.SetLocal(j, i, 1.0);

    DistMatrixVCSType A_vcs = A_stst;
    skylark::base::detail::outer_panel_mixed_gemm_impl_tn(
            -1.0, B, A_vcs, 1.5, result_stst);
    check_matrix(result_stst, tn_expected,
                 "Result of outer panel TN gemm not as expected");

    if(world.rank() == 0)
        std::cout << "outer panel: OK" << std::endl;

    if(world.rank() == 0)
        std::cout << "Testing CombBLAS x Elemental (*/*) = Elemental (VX/*) :";

    DistMatrixVCSType result_vcs(matrix_size, matrix_size, grid);
    for( size_t j = 0; j < result_vcs.LocalHeight(); j++ )
        for( size_t i = 0; i < result_vcs.LocalWidth(); i++ )
            result_vcs.SetLocal(j, i, 1.0);

    skylark::base::detail::outer_panel_mixed_gemm_impl_nn(
            -1.0, B, A_stst, 1.5, result_vcs);
    check_matrix(result_vcs, nn_expected,
                 "Result of outer panel NN gemm not as expected");

    if(world.rank() == 0)
        std::cout << "outer panel: OK" << std::endl;

    for( size_t j = 0; j < result_vcs.LocalHeight(); j++ )
        for( size_t i = 0; i < result_vcs.LocalWidth(); i++ )
            result_vcs.SetLocal(j, i, 1.0);

    skylark::base::detail::inner_panel_mixed_gemm_impl_nn(
            -1.0, B, A_stst, 1.5, result_vcs);
    check_matrix(result_vcs, nn_expected,
                 "Result of inner panel NN gemm not as expected");

    if(world.rank() == 0)
        std::cout << "inner panel: OK" << std::endl;

    elem::Finalize();

    return 0;
}

