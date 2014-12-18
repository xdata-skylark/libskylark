#include <skylark.hpp>
#include <boost/mpi.hpp>
#include <El.hpp>
#include <iostream>
#include "../base/QR.hpp"
#include <cfloat>
#include <vector>


/** Aliases for matrix types */
typedef El::DistMatrix<double> dist_matrix_t;
typedef El::Matrix<double> matrix_t;
typedef El::DistMatrix<double, El::VR, El::STAR> vr_star_dist_matrix_t;
typedef El::DistMatrix<double, El::STAR, El::STAR> star_star_matrix_t;
typedef skylark::sketch::JLT_t<dist_matrix_t, dist_matrix_t> sketch_transform_t;

using namespace std;

int main(int argc, char* argv[]) {

   /** Initialize MPI  */
#ifdef SKYLARK_HAVE_OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    MPI_Comm mpi_world(world);
    El::Grid grid(mpi_world);

    /** Initialize Elemental */
    El::Initialize (argc, argv);

    /** Initialize context */
    skylark::base::context_t context(0);

    /** Declare matrices */
    dist_matrix_t A(grid), B(grid), C(grid);
    El::Uniform(B, 5000, 100);
    skylark::base::qr::ExplicitUnitary(B);

    El::Uniform(C, 100, 100);
    skylark::base::qr::ExplicitUnitary(C);

    //star_star_matrix_t S(100,100);
    dist_matrix_t S(100,100);
    El::Zero(S);
	
    vector<double> diag(100);

    for( int j=0; j<100; ++j )
    {
	diag[j] = exp(-j)*100;
        std::cout << exp(-j) *100 << "\n";
    }

    El::Diagonal(S, diag);
    dist_matrix_t tmp(grid);

    El::Gemm(El::NORMAL, El::NORMAL, double(1), B, S, tmp);
    El::Gemm(El::NORMAL, El::ADJOINT, double(1), tmp, C, A);

    dist_matrix_t U(grid), V(grid);
    vr_star_dist_matrix_t S1;

    dist_matrix_t A1(A);
    El::SVD(A1,S1,V);

    El::Print(S1, "S1");

    /** Declare matrices */
    dist_matrix_t A2(A);
    dist_matrix_t U1(grid), V1(grid);
    vr_star_dist_matrix_t S2;

    int sketch_size	=	50;
    int target_rank	=	10;

    skylark::nla::rand_svd_params_t params(sketch_size-target_rank);

    skylark::nla::randsvd_t<skylark::sketch::JLT_t> rand_svd;
    rand_svd(A2, target_rank, U1, S2, V1, params, context);

    El::Print(S2, "S2");

    El::Finalize();
    return 0;
}
