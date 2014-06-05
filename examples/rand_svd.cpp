#include <skylark.hpp>
#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <iostream>
#include "../base/QR.hpp"
#include <cfloat>
#include <vector>


/** Aliases for matrix types */
typedef elem::DistMatrix<double> dist_matrix_t;
typedef elem::Matrix<double> matrix_t;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> vr_star_dist_matrix_t;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> star_star_matrix_t;
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
    elem::Grid grid(mpi_world);

    /** Initialize Elemental */
    elem::Initialize (argc, argv);

    /** Initialize context */
    skylark::base::context_t context(0);

    /** Declare matrices */
    dist_matrix_t A(grid), B(grid), C(grid);
    elem::Uniform(B, 5000, 100);
    skylark::base::qr::Explicit(B);

    elem::Uniform(C, 100, 100);
    skylark::base::qr::Explicit(C);

    //star_star_matrix_t S(100,100);
    dist_matrix_t S(100,100);
    elem::Zero(S);
	
    vector<double> diag(100);

    for( int j=0; j<100; ++j )
    {
	diag[j] = exp(-j)*100;
        std::cout << exp(-j) *100 << "\n";
    }

    elem::Diagonal(S, diag);
    dist_matrix_t tmp(grid);

    elem::Gemm(elem::NORMAL, elem::NORMAL, double(1), B, S, tmp);
    elem::Gemm(elem::NORMAL, elem::ADJOINT, double(1), tmp, C, A);

    dist_matrix_t U(grid), V(grid);
    vr_star_dist_matrix_t S1;

    dist_matrix_t A1(A);
    elem::SVD(A1,S1,V);

    elem::Print(S1, "S1");

    /** Declare matrices */
    dist_matrix_t A2(A);
    dist_matrix_t U1(grid), V1(grid);
    vr_star_dist_matrix_t S2;

    int sketch_size	=	50;
    int target_rank	=	10;

    skylark::nla::rand_svd_params_t params(sketch_size-target_rank);

    skylark::nla::randsvd_t<skylark::sketch::JLT_t> rand_svd;
    rand_svd(A2, target_rank, U1, S2, V1, params, context);

    elem::Print(S2, "S2");

    elem::Finalize();
    return 0;
}
