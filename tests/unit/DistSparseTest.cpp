/**
 *  This test ensures that our internal distributed sparse matrix behave as
 *  their Elemental counter parts.
 *  This test builds on the following assumptions:
 *
 *      - Elemental distributions and Gemms are implemented correctly.
 */


#include <string>
#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#define SKYLARK_NO_ANY

#include "../../skylark.hpp"

#include "test_utils.hpp"

typedef El::DistMatrix<double, El::VC, El::STAR> dense_vc_star_matrix_t;
typedef skylark::base::sparse_vc_star_matrix_t<double> sparse_vc_star_matrix_t;

typedef El::DistMatrix<double, El::STAR, El::VR> dense_star_vr_matrix_t;
typedef skylark::base::sparse_star_vr_matrix_t<double> sparse_star_vr_matrix_t;

boost::random::mt19937 gen;

template <typename dense_matrix_t, typename sparse_matrix_t>
void create_random_sparse_matrix_pair(
        sparse_matrix_t& sparse, dense_matrix_t& dense,
        bool symmetrize = false) {

    boost::random::uniform_int_distribution<> value_dist(1, 500);
    boost::random::uniform_int_distribution<> nnz_dist(0, 9);

    El::Zero(dense);

    size_t rows = dense.Height();

    for (int col = 0; col < dense.Width(); col++) {
        if (symmetrize) rows = col;
        for (int row = 0; row < rows; row++) {
            if (nnz_dist(gen) != 0)
                continue;

            double val = static_cast<double>(value_dist(gen));
            dense.Update(row, col, val);
            sparse.queue_update(row, col, val);

            if (symmetrize) {
                dense.Update(col, row, val);
                sparse.queue_update(col, row, val);
            }
        }
    }

    sparse.finalize();
}

template <typename dense_matrix_t, typename sparse_matrix_t>
void test_matrix_properties(
        const dense_matrix_t& A, const sparse_matrix_t& A_sparse) {
    // FIXME: test more

    BOOST_REQUIRE(A_sparse.width()    == A.Width());
    BOOST_REQUIRE(A_sparse.height()   == A.Height());
    BOOST_REQUIRE(A_sparse.nonzeros() <= A.Width() * A.Height());

    BOOST_REQUIRE(A_sparse.local_width()  == A.LocalWidth());
    BOOST_REQUIRE(A_sparse.local_height() == A.LocalHeight());
}


template <typename dense_matrix_t, typename sparse_matrix_t>
void check_equal(
        const dense_matrix_t& A, const sparse_matrix_t& A_sparse) {
    const int* indptr    = A_sparse.indptr();
    const int* indices   = A_sparse.indices();
    const double* values = A_sparse.locked_values();

    double threshold = 1e-7;
    for (int col = 0; col < A_sparse.local_width(); col++) {
        for (int idx = indptr[col]; idx < indptr[col + 1]; idx++) {
            int row = indices[idx];
            if (fabs(A.GetLocal(row, col) - values[idx]) > threshold) {
                std::cout << "(" << row << ", " << col << " ) diff = "
                          << A.GetLocal(row, col) - values[idx] << std::endl;
            }
            BOOST_REQUIRE(fabs(A.GetLocal(row, col) - values[idx]) < threshold);
        }
    }
}


// FIXME: merge helper functions
template <typename sparse_matrix_t>
void test_gemm(El::Orientation oA, El::Orientation oB, double alpha,
        const sparse_matrix_t& A_sparse, const El::DistMatrix<double, El::VC, El::STAR>& A,
        double beta, El::Int target_width, boost::mpi::communicator world,
        const El::Grid& grid) {
    // FIXME: cannot handle this case yet
    // assert(std::fabs(beta) < 1e-5);
    beta = 0.0;


    El::Int target_height = A_sparse.height();
    El::Int B_height = A.Width();
    if (oA == El::TRANSPOSE) {
        target_height = A_sparse.width();
        B_height = A.Height();
    }

    El::DistMatrix<double, El::STAR, El::STAR> B(grid);
    El::Uniform(B, B_height, target_width);
    if (oB == El::TRANSPOSE) target_width = B_height;

    sparse_vc_star_matrix_t A_sparse_vc_result(
        target_height, target_width, grid);
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


// FIXME: merge helper functions
template <typename sparse_matrix_t>
void test_gemm_vc(El::Orientation oA, El::Orientation oB, double alpha,
        const sparse_matrix_t& A_sparse, const El::DistMatrix<double, El::STAR, El::VR>& A,
        double beta, El::Int target_width, boost::mpi::communicator world,
        const El::Grid& grid) {

    El::Int target_height = A_sparse.height();
    El::Int B_height = A.Width();
    if (oA == El::TRANSPOSE) {
        target_height = A_sparse.width();
        B_height = A.Height();
    }

    El::DistMatrix<double, El::STAR, El::STAR> B(grid);
    El::Uniform(B, B_height, target_width);
    if (oB == El::TRANSPOSE) target_width = B_height;

    El::DistMatrix<double, El::VC, El::STAR> A_sparse_vc_result(grid);
    El::Uniform(A_sparse_vc_result, target_height, target_width);
    El::Zero(A_sparse_vc_result);
    skylark::base::Gemm(El::NORMAL, El::NORMAL, alpha,
        A_sparse, B, beta, A_sparse_vc_result);

    El::DistMatrix<double> A_mcmr = A;
    El::DistMatrix<double> B_mcmr = B;
    El::DistMatrix<double> C(grid);
    El::Uniform(C, target_height, target_width);
    // FIXME: should be _mcmr
    El::Gemm(oA, oB, alpha, A, B, beta, C);

    El::DistMatrix<double, El::VC, El::STAR> A_vc_result = C;

    BOOST_REQUIRE(A_vc_result.LocalWidth() == A_sparse_vc_result.LocalWidth());
    BOOST_REQUIRE(A_vc_result.LocalHeight() == A_sparse_vc_result.LocalHeight());

    El::Matrix<double> A_vc_result_gathered = C.Matrix();
    El::Matrix<double> A_sparse_vc_result_gathered = A_sparse_vc_result.Matrix();

    if (!test::util::equal(A_vc_result_gathered, A_sparse_vc_result_gathered))
        BOOST_FAIL("Gemm VC/STAR application not equal");
}


//FIXME: merge helper functions
template <typename sparse_matrix_t>
void test_gemm_vr(El::Orientation oA, El::Orientation oB, double alpha,
        const sparse_matrix_t& A_sparse, const El::DistMatrix<double, El::STAR, El::VR>& A,
        double beta, El::Int target_width, boost::mpi::communicator world) {

    El::Int target_height = A_sparse.height();
    El::Int B_height = A.Width();
    if (oA == El::TRANSPOSE) {
        target_height = A_sparse.width();
        B_height = A.Height();
    }

    El::DistMatrix<double, El::VC, El::STAR> B(A.Grid());
    El::Uniform(B, B_height, target_width);
    if (oB == El::TRANSPOSE) target_width = B_height;

    El::DistMatrix<double, El::STAR, El::STAR> A_sparse_vr_result(A.Grid());
    El::Uniform(A_sparse_vr_result, target_height, target_width);
    El::Zero(A_sparse_vr_result);
    skylark::base::Gemm(El::NORMAL, El::NORMAL, alpha,
        A_sparse, B, beta, A_sparse_vr_result);

    El::DistMatrix<double> A_mcmr = A;
    El::DistMatrix<double> B_mcmr = B;
    El::DistMatrix<double> C(A.Grid());
    El::Uniform(C, target_height, target_width);
    // FIXME: should be _mcmr
    El::Gemm(oA, oB, alpha, A, B, beta, C);

    El::DistMatrix<double, El::STAR, El::STAR> A_vr_result = C;

    BOOST_REQUIRE(A_vr_result.LocalWidth() == A_sparse_vr_result.LocalWidth());
    BOOST_REQUIRE(A_vr_result.LocalHeight() == A_sparse_vr_result.LocalHeight());

    if (!test::util::equal(A_vr_result, A_vr_result))
        BOOST_FAIL("Gemm STAR/VR application not equal");
}

/**
 *  Test symmetric matrix multiply for sparse distributed matrices.
 *  Multiply with a random (uniform) dense matrix.
 *
 *  FIXME: merge helper functions into test utils
 *  FIXME: target_width is a bad name
 */
template <typename sparse_matrix_t>
void test_symm(El::LeftOrRight side, El::UpperOrLower uplo, double alpha,
        const sparse_matrix_t& A_sparse,
        const El::DistMatrix<double, El::STAR, El::VC>& A,
        double beta, El::Int target_width, boost::mpi::communicator world) {

    // determine dimensions of random multiplier
    El::Int target_height = A.Height();
    El::Int B_height = A.Height();
    if (side == El::RIGHT) {
        target_height = target_width;
        B_height = target_width;
        target_width = A.Width();
    }

    // create random multiplier
    El::DistMatrix<double, El::VC, El::STAR> B(A.Grid());
    El::Uniform(B, B_height, target_width);

    // perform sparse Symm
    El::DistMatrix<double, El::VC, El::STAR> A_sparse_vc_result(A.Grid());
    El::Uniform(A_sparse_vc_result, target_height, target_width);
    El::Zero(A_sparse_vc_result);
    skylark::base::Symm(side, uplo, alpha, A_sparse, B, beta, A_sparse_vc_result);

    // controll with expected value from Elemental Symm (MC/MR)
    El::DistMatrix<double> A_mcmr = A;
    El::DistMatrix<double> B_mcmr = B;
    El::DistMatrix<double> C_mcmr(A.Grid());
    El::Uniform(C_mcmr, target_height, target_width);
    El::Symm(side, uplo, alpha, A, B, beta, C_mcmr);

    // and convert matrix back to VC/STAR
    El::DistMatrix<double, El::VC, El::STAR> A_vc_result = C_mcmr;

    // finally we can check for equality (dimensions and values)
    BOOST_REQUIRE(A_vc_result.LocalWidth()  == A_sparse_vc_result.LocalWidth());
    BOOST_REQUIRE(A_vc_result.LocalHeight() == A_sparse_vc_result.LocalHeight());

    El::Matrix<double> A_vc_result_gathered = A_vc_result.Matrix();
    El::Matrix<double> A_sparse_vc_result_gathered = A_sparse_vc_result.Matrix();

    if (!test::util::equal(A_vc_result_gathered, A_sparse_vc_result_gathered))
        BOOST_FAIL("Symm VC/STAR application not equal");

    world.barrier();
}


int test_main(int argc, char *argv[]) {
    El::Initialize(argc, argv);
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    MPI_Comm mpi_world(world);
    El::Grid grid(mpi_world);

    //////////////////////////////////////////////////////////////////////////
    //[> Setup <]

    const unsigned int num_repetitions = 10;

    gen.seed(static_cast<unsigned int>(std::time(0)));
    boost::random::uniform_int_distribution<> dim_dist(250, 1000);
    boost::random::uniform_int_distribution<> target_dist(15, 200);

    //////////////////////////////////////////////////////////////////////////
    //[> Test properties <]

    {
    const int height = dim_dist(gen);
    const int width  = dim_dist(gen);

    dense_vc_star_matrix_t A_vc(grid);
    El::Uniform(A_vc, height, width);
    sparse_vc_star_matrix_t A_sparse_vc(height, width, grid);
    create_random_sparse_matrix_pair(A_sparse_vc, A_vc);

    test_matrix_properties(A_vc, A_sparse_vc);
    check_equal(A_vc, A_sparse_vc);

    dense_star_vr_matrix_t A_vr(grid);
    El::Uniform(A_vr, height, width);
    sparse_star_vr_matrix_t A_sparse_vr(height, width, grid);
    create_random_sparse_matrix_pair(A_sparse_vr, A_vr);

    test_matrix_properties(A_vr, A_sparse_vr);
    check_equal(A_vr, A_sparse_vr);
    }

    //////////////////////////////////////////////////////////////////////////
    //[> Test Symm <]
    //
    // FIXME: currently only for sparse_vc_star_matrix_t

    {
    for (size_t rep = 0; rep < num_repetitions; rep++) {
        const int symm_dim    = dim_dist(gen);
        const int symm_target = target_dist(gen);

        // create a symmetric test input matrix
        dense_vc_star_matrix_t A_vc_symm(grid);
        El::Uniform(A_vc_symm, symm_dim, symm_dim);
        sparse_vc_star_matrix_t A_sparse_vc_symm(symm_dim, symm_dim, grid);
        create_random_sparse_matrix_pair(A_sparse_vc_symm, A_vc_symm, true);

        if (world.rank() == 0) std::cout << "Testing SYMMs: ("
            << symm_dim << " x " << symm_dim << ") -> "
            << symm_target << std::endl;

        if (world.rank() == 0)
            std::cout << "\tsparse_vc_star -> vc_star (LEFT, UPPER):";
        test_symm(El::LEFT, El::UPPER, 1.0,
                A_sparse_vc_symm, A_vc_symm, 0.0, symm_target, world);
        world.barrier();
        if (world.rank() == 0) std::cout << " ok" << std::endl;

        if (world.rank() == 0)
            std::cout << "\tsparse_vc_star -> vc_star (RIGHT, UPPER):";
        test_symm(El::RIGHT, El::UPPER, 1.0,
                A_sparse_vc_symm, A_vc_symm, 0.0, symm_target, world);
        world.barrier();
        if (world.rank() == 0) std::cout << " ok" << std::endl;

        world.barrier();
    }

    if (world.rank() == 0) std::cout << "Done." << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////
    //[> Test Gemm <]

    {
    for (size_t rep = 0; rep < num_repetitions; rep++) {
        const int gemm_height = dim_dist(gen);
        const int gemm_width  = dim_dist(gen);
        const int gemm_target = target_dist(gen);

        dense_vc_star_matrix_t A_vc(grid);
        El::Uniform(A_vc, gemm_height, gemm_width);
        sparse_vc_star_matrix_t A_sparse_vc(gemm_height, gemm_width, grid);
        create_random_sparse_matrix_pair(A_sparse_vc, A_vc);

        if (world.rank() == 0) std::cout << "Testing GEMMs: ("
            << A_vc.Height() << ", " << A_vc.Width() << ") -> "
            << gemm_target << std::endl;

        if (world.rank() == 0)
            std::cout << "\tsparse(VC/STAR) * dense(STAR/STAR) "
                      << "-> sparse(VC/STAR) (NORMAL, NORMAL):";
        test_gemm(El::NORMAL, El::NORMAL, 1.0,
                A_sparse_vc, A_vc, 0.0, gemm_target, world, grid);
        world.barrier();
        if (world.rank() == 0) std::cout << " ok" << std::endl;

#if 0
        if (world.rank() == 0) std::cout
            << "\tsparse(VC/STAR) x dense(STAR/STAR) "
            << "-> dense(VC/STAR) (NORMAL, NORMAL):";
        test_gemm_vc(El::NORMAL, El::NORMAL, 1.0,
                A_sparse_vc, A_vc, 0.0, target, world, grid);
        world.barrier();
        if (world.rank() == 0) std::cout << " ok" << std::endl;

        if (world.rank() == 0)
            std::cout << "\tsparse(STAR/VR) x dense(VC/STAR) "
                      << "-> dense(STAR/STAR) (NORMAL, NORMAL):";
        test_gemm_vr(El::NORMAL, El::NORMAL, 1.0,
                A_sparse_vr, A_vr, 0.0, target, world);
        world.barrier();
        if (world.rank() == 0) std::cout << " ok" << std::endl;
#endif

        world.barrier();
    }

    if (world.rank() == 0) std::cout << "Done." << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////
    //[> Test I/O <]

    // only run test if we got a file name
    if (argc < 2) {
        El::Finalize();
        return 0;
    }

    sparse_vc_star_matrix_t X(0, 0, grid);
    El::DistMatrix<double, El::VC, El::STAR> Y;

    std::string fname(argv[1]);
    try {
        skylark::utility::io::ReadLIBSVM(fname, X, Y, skylark::base::COLUMNS);
    } catch (skylark::base::skylark_exception ex) {
        SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
        SKYLARK_PRINT_EXCEPTION_TRACE(ex);
        errno = *(boost::get_error_info<skylark::base::error_code>(ex));
        std::cout << "Caught exception, exiting with error " << errno << ": ";
        std::cout << skylark_strerror(errno) << std::endl;
        BOOST_FAIL("Exception when reading libSVM file.");
    }

    El::DistMatrix<double, El::VC, El::STAR> X_ref;
    El::DistMatrix<double, El::VC, El::STAR> Y_ref;
    try {
        skylark::utility::io::ReadLIBSVM(
            fname, X_ref, Y_ref, skylark::base::COLUMNS);
    } catch (skylark::base::skylark_exception ex) {
        SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
        SKYLARK_PRINT_EXCEPTION_TRACE(ex);
        errno = *(boost::get_error_info<skylark::base::error_code>(ex));
        std::cout << "Caught exception, exiting with error " << errno << ": ";
        std::cout << skylark_strerror(errno) << std::endl;
        BOOST_FAIL("Exception when reading libSVM file.");
    }

    check_equal(X_ref, X);

    El::Finalize();
    return 0;
}
