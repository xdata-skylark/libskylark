#include <skylark.hpp>
#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <iostream>

/** Aliases for matrix types */
typedef elem::DistMatrix<double> dist_matrix_t;
typedef elem::Matrix<double> matrix_t;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> vr_star_dist_matrix_t;
typedef skylark::sketch::JLT_t<dist_matrix_t, dist_matrix_t> sketch_transform_t;

int main(int argc, char* argv[]) {

   /** Initialize MPI  */
#ifdef SKYLARK_HAVE_OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    MPI_Comm mpi_world(world);
    elem::Grid grid(mpi_world);

    /** Initialize Elemental */
    elem::Initialize (argc, argv);

    /** Example parameters */
    int height = 10;
    int width  = 10;
    int sketch_size = 6;
    int target_rank = 4;
    int num_iterations = 2;


    /** Initialize context */
    skylark::base::context_t context(0);

#if  0
    /** Declare matrices */
    dist_matrix_t A(grid);
    dist_matrix_t B(grid);
    dist_matrix_t C(grid);
    elem::Uniform(B, 5000, 10);
    elem::Uniform(C, 10, 100);
    elem::Gemm(elem::NORMAL, elem::NORMAL, double(1), B, C, A);
    dist_matrix_t U(grid), V(grid);
    vr_star_dist_matrix_t S(grid);

    dist_matrix_t A1(A);
    elem::SVD(A1,S,V);

    //skylark::nla::RandSVD(A, target_rank, U, S, V, params, context);
    /** Print the resulting matrices: U, S, V */
    //elem::Print(U, "U");
    elem::Print(S, "S");
    //elem::Print(V, "V");
#endif

#if  0
    /** Declare matrices */
    dist_matrix_t A2(A);
    //elem::Uniform(A, height, width);
    dist_matrix_t U1(grid), V1(grid);
    vr_star_dist_matrix_t S1(grid);

    sketch_size	=	50;
    target_rank	=	10;

    skylark::nla::rand_svd_params_t params(sketch_size-target_rank);

    skylark::nla::randsvd_t<skylark::sketch::JLT_t> rand_svd;
    rand_svd(A2, target_rank, U1, S1, V1, params, context);

    //skylark::nla::RandSVD(A, target_rank, U, S, V, params, context);
    /** Print the resulting matrices: U, S, V */
    //elem::Print(U, "U");
    elem::Print(S1, "S");
    //elem::Print(V, "V");
#endif

#if 1
    vr_star_dist_matrix_t A3(grid);
    elem::Uniform(A3, height, width);
    matrix_t U3, V3;
    matrix_t S3;

    skylark::nla::rand_svd_params_t params(sketch_size-target_rank);

    skylark::nla::randsvd_t<skylark::sketch::FJLT_t> rand_svd;
    rand_svd(A3, target_rank, U3, S3, V3, params, context);

    /** Print the resulting matrices: U, S, V */
    elem::Print(U, "U");
    elem::Print(S, "S");
    elem::Print(V, "V");
#endif

#if 0
    /** Declare matrices */
    dist_matrix_t A(grid);
    elem::Uniform(A, height, width);
    dist_matrix_t U(grid), V(grid);
    vr_star_dist_matrix_t S(grid);

    skylark::nla::RandSVD<dist_matrix_t, dist_matrix_t, vr_star_dist_matrix_t, skylark::sketch::FJLT_t, skylark::sketch::rowwise_tag> 			(A, target_rank, width, sketch_size, U, S, V, skylark::nla::SUBSPACE_ITERATIONS, num_iterations, context);

    /** Print the resulting matrices: U, S, V */
    elem::Print(U, "U");
    elem::Print(S, "S");
    elem::Print(V, "V");
#endif

#if 0
    /** Declare matrices */
    dist_matrix_t A(grid);
    elem::Uniform(A, height, width);
    dist_matrix_t U(grid), V(grid);
    vr_star_dist_matrix_t S(grid);

    skylark::nla::RandSVD<dist_matrix_t, dist_matrix_t, vr_star_dist_matrix_t, skylark::sketch::CWT_t, skylark::sketch::rowwise_tag> 			(A, target_rank, width, sketch_size, U, S, V, skylark::nla::SUBSPACE_ITERATIONS, num_iterations, context);

    /** Print the resulting matrices: U, S, V */
    elem::Print(U, "U");
    elem::Print(S, "S");
    elem::Print(V, "V");
#endif

    elem::Finalize();
    return 0;
}
