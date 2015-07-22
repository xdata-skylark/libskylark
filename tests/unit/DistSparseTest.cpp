/**
 *  This test ensures that our internal distributed sparse matrix behave as
 *  their Elemental counter parts.
 *  This test builds on the following assumptions:
 *
 *      - Elemental distributions and Gemms are implemented correctly.
 */


#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#define SKYLARK_NO_ANY

#include "../../skylark.hpp"

#include "test_utils.hpp"

typedef El::DistMatrix<double, El::VC, El::STAR> dense_vc_star_matrix_t;
typedef skylark::base::sparse_vc_star_matrix_t<double> sparse_vc_star_matrix_t;

typedef El::DistMatrix<double, El::STAR, El::VR> dense_star_vr_matrix_t;
typedef skylark::base::sparse_star_vr_matrix_t<double> sparse_star_vr_matrix_t;


template <typename dense_matrix_t, typename sparse_matrix_t>
void test_matrix_properties(
        const dense_matrix_t& A, const sparse_matrix_t& A_sparse) {

    //FIXME: test more

    BOOST_REQUIRE(A_sparse.width()    == A.Width());
    BOOST_REQUIRE(A_sparse.height()   == A.Height());
    BOOST_REQUIRE(A_sparse.nonzeros() == A.Width() * A.Height());

    BOOST_REQUIRE(A_sparse.local_width()  == A.LocalWidth());
    BOOST_REQUIRE(A_sparse.local_height() == A.LocalHeight());
}


template <typename dense_matrix_t, typename sparse_matrix_t>
void check_equal(
        const dense_matrix_t& A, const sparse_matrix_t& A_sparse) {

    const int* indptr    = A_sparse.indptr();
    const int* indices   = A_sparse.indices();
    const double* values = A_sparse.locked_values();

    for(int col = 0; col < A_sparse.local_width(); col++) {
        for(int idx = indptr[col]; idx < indptr[col + 1]; idx++) {

            int row = indices[idx];
            BOOST_REQUIRE(fabs(A.GetLocal(row, col) - values[idx]) < 1e-8);
        }
    }
}


//FIXME: merge helper functions
template <typename sparse_matrix_t>
void test_gemm(El::Orientation oA, El::Orientation oB, double alpha,
        const sparse_matrix_t& A_sparse, const El::DistMatrix<double, El::VC, El::STAR>& A,
        double beta, El::Int target_width, boost::mpi::communicator world,
        const El::Grid& grid) {

    //FIXME: cannot handle this case yet
    //assert(std::fabs(beta) < 1e-5);
    beta = 0.0;


    El::Int target_height = A_sparse.height();
    El::Int B_height = A.Width();
    if(oA == El::TRANSPOSE) {
        target_height = A_sparse.width();
        B_height = A.Height();
    }

    El::DistMatrix<double, El::STAR, El::STAR> B(grid);
    El::Uniform(B, B_height, target_width);
    if(oB == El::TRANSPOSE) target_width = B_height;

    sparse_vc_star_matrix_t A_sparse_vc_result(
        target_height, target_width, world, grid);
    skylark::base::Gemm(El::NORMAL, El::NORMAL, alpha,
        A_sparse, B, beta, A_sparse_vc_result);

    BOOST_REQUIRE(A_sparse_vc_result.is_finalized());

    El::DistMatrix<double> A_mcmr = A;
    El::DistMatrix<double> B_mcmr = B;
    El::DistMatrix<double> C(grid);
    El::Uniform(C, target_height, target_width);
    El::Gemm(oA, oB, alpha, A, B, beta, C);

    dense_vc_star_matrix_t A_vc_result = C;

    BOOST_REQUIRE(A_vc_result.LocalWidth() == A_sparse_vc_result.local_width());
    BOOST_REQUIRE(A_vc_result.LocalHeight() == A_sparse_vc_result.local_height());

    check_equal(A_vc_result, A_sparse_vc_result);
}


//FIXME: merge helper functions
template <typename sparse_matrix_t>
void test_gemm_vc(El::Orientation oA, El::Orientation oB, double alpha,
        const sparse_matrix_t& A_sparse, const El::DistMatrix<double, El::STAR, El::VR>& A,
        double beta, El::Int target_width, boost::mpi::communicator world,
        const El::Grid& grid) {

    El::Int target_height = A_sparse.height();
    El::Int B_height = A.Width();
    if(oA == El::TRANSPOSE) {
        target_height = A_sparse.width();
        B_height = A.Height();
    }

    El::DistMatrix<double, El::STAR, El::STAR> B(grid);
    El::Uniform(B, B_height, target_width);
    if(oB == El::TRANSPOSE) target_width = B_height;

    El::DistMatrix<double, El::VC, El::STAR> A_sparse_vc_result(grid);
    El::Uniform(A_sparse_vc_result, target_height, target_width);
    El::Zero(A_sparse_vc_result);
    skylark::base::Gemm(El::NORMAL, El::NORMAL, alpha,
        A_sparse, B, beta, A_sparse_vc_result);

    El::DistMatrix<double> A_mcmr = A;
    El::DistMatrix<double> B_mcmr = B;
    El::DistMatrix<double> C(grid);
    El::Uniform(C, target_height, target_width);
    El::Gemm(oA, oB, alpha, A, B, beta, C);

    El::DistMatrix<double, El::VC, El::STAR> A_vc_result = C;

    BOOST_REQUIRE(A_vc_result.LocalWidth() == A_sparse_vc_result.LocalWidth());
    BOOST_REQUIRE(A_vc_result.LocalHeight() == A_sparse_vc_result.LocalHeight());

    El::Matrix<double> A_vc_result_gathered = C.Matrix();
    El::Matrix<double> A_sparse_vc_result_gathered = A_sparse_vc_result.Matrix();

    if (!equal(A_vc_result_gathered, A_sparse_vc_result_gathered))
        BOOST_FAIL("Gemm VC/STAR application not equal");
}


//FIXME: merge helper functions
template <typename sparse_matrix_t>
void test_gemm_vr(El::Orientation oA, El::Orientation oB, double alpha,
        const sparse_matrix_t& A_sparse, const El::DistMatrix<double, El::STAR, El::VR>& A,
        double beta, El::Int target_width, boost::mpi::communicator world,
        const El::Grid& grid) {

    El::Int target_height = A_sparse.height();
    El::Int B_height = A.Width();
    if(oA == El::TRANSPOSE) {
        target_height = A_sparse.width();
        B_height = A.Height();
    }

    El::DistMatrix<double, El::VC, El::STAR> B(grid);
    El::Uniform(B, B_height, target_width);
    if(oB == El::TRANSPOSE) target_width = B_height;

    El::DistMatrix<double, El::STAR, El::STAR> A_sparse_vr_result(grid);
    El::Uniform(A_sparse_vr_result, target_height, target_width);
    El::Zero(A_sparse_vr_result);
    skylark::base::Gemm(El::NORMAL, El::NORMAL, alpha,
        A_sparse, B, beta, A_sparse_vr_result);

    El::DistMatrix<double> A_mcmr = A;
    El::DistMatrix<double> B_mcmr = B;
    El::DistMatrix<double> C(grid);
    El::Uniform(C, target_height, target_width);
    El::Gemm(oA, oB, alpha, A, B, beta, C);

    El::DistMatrix<double, El::STAR, El::STAR> A_vr_result = C;

    BOOST_REQUIRE(A_vr_result.LocalWidth() == A_sparse_vr_result.LocalWidth());
    BOOST_REQUIRE(A_vr_result.LocalHeight() == A_sparse_vr_result.LocalHeight());

    if (!equal(A_vr_result, A_vr_result))
        BOOST_FAIL("Gemm STAR/VR application not equal");
}


int test_main(int argc, char *argv[]) {

    /** Initialize Elemental */
    El::Initialize (argc, argv);

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    MPI_Comm mpi_world(world);
    El::Grid grid(mpi_world);


    //////////////////////////////////////////////////////////////////////////
    //[> Parameters <]

    //FIXME: use random sizes?
    const int height = 20;
    const int width  = 10;

    //////////////////////////////////////////////////////////////////////////
    //[> Setup test <]

    dense_vc_star_matrix_t A_vc(grid);
    El::Uniform(A_vc, height, width);
    El::Zero(A_vc);

    sparse_vc_star_matrix_t A_sparse_vc(height, width, world, grid);

    double count = 0.0;
    for(int col = 0; col < A_vc.Width(); col++) {
        for(int row = 0; row < A_vc.Height(); row++) {
            A_vc.Update(row, col, count);
            A_sparse_vc.queue_update(row, col, count);
            count++;
        }
    }

    A_sparse_vc.finalize();


    dense_star_vr_matrix_t A_vr(grid);
    El::Uniform(A_vr, height, width);
    El::Zero(A_vr);

    sparse_star_vr_matrix_t A_sparse_vr(height, width, world, grid);

    count = 0.0;
    for(int col = 0; col < A_vr.Width(); col++) {
        for(int row = 0; row < A_vr.Height(); row++) {
            A_vr.Update(row, col, count);
            A_sparse_vr.queue_update(row, col, count);
            count++;
        }
    }

    A_sparse_vr.finalize();


    //////////////////////////////////////////////////////////////////////////
    //[> Test properties <]

    test_matrix_properties(A_vc, A_sparse_vc);
    test_matrix_properties(A_vr, A_sparse_vr);

    check_equal(A_vc, A_sparse_vc);
    check_equal(A_vr, A_sparse_vr);


    //////////////////////////////////////////////////////////////////////////
    //[> Test Gemm <]

    const int target = 5;

    if(world.rank() == 0) std::cout << "sparse_vc_star -> sparse_vc_star (NORMAL, NORMAL)" << std::endl;
    test_gemm(El::NORMAL, El::NORMAL, 1.0,
            A_sparse_vc, A_vc, 0.0, target, world, grid);
    world.barrier();

    if(world.rank() == 0) std::cout << "sparse_vc_star -> dense_vc_star (NORMAL, NORMAL)" << std::endl;
    test_gemm_vc(El::NORMAL, El::NORMAL, 1.0,
            A_sparse_vc, A_vc, 0.0, target, world, grid);
    world.barrier();

    if(world.rank() == 0) std::cout << "sparse_star_vr -> dense_star_star (NORMAL, NORMAL)" << std::endl;
    test_gemm_vr(El::NORMAL, El::NORMAL, 1.0,
            A_sparse_vr, A_vr, 0.0, target, world, grid);
    world.barrier();


    //////////////////////////////////////////////////////////////////////////
    //[> Test I/O <]

    sparse_vc_star_matrix_t X(0, 0, world, grid);
    El::DistMatrix<double, El::VC, El::STAR> Y;
    std::string fname(argv[1]);
    skylark::utility::io::ReadLIBSVM(fname, X, Y, skylark::base::COLUMNS);

    El::DistMatrix<double, El::VC, El::STAR> X_ref;
    El::DistMatrix<double, El::VC, El::STAR> Y_ref;
    skylark::utility::io::ReadLIBSVM(fname, X_ref, Y_ref, skylark::base::COLUMNS);

    check_equal(X_ref, X);

    El::Finalize();

    return 0;
}
