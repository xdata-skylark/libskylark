#include <skylark.hpp>
#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <iostream>
#include "../base/QR.hpp"
#include <cfloat>


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

#if  1
    /** Declare matrices */
    matrix_t A;
    matrix_t B;
    matrix_t C;
    elem::Uniform(B, 5000, 5000);
    skylark::base::qr::Explicit(B);

    elem::Uniform(C, 10, 10);
    skylark::base::qr::Explicit(C);

    matrix_t S(5000,10);
    S.Empty();

    for( int j=0; j<10; ++j )
    {
     S.Set( j, j, exp(-j)*100);    
     std::cout << exp(-j) *100 << "\n";
    }

    matrix_t tmp;

    elem::Gemm(elem::NORMAL, elem::NORMAL, double(1), B, S, tmp);
    elem::Gemm(elem::NORMAL, elem::ADJOINT, double(1), tmp, C, A);


    matrix_t U, V, S1;

    matrix_t A1(A);
    elem::SVD(A1,S,V);

    elem::Print(S, "S");
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

    elem::Finalize();
    return 0;
}
