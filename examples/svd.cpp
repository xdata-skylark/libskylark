#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();

    skylark::base::context_t context(38734);

    El::DistMatrix<double> A;
    El::DistMatrix<double> U, S, V, Y;

   boost::mpi::timer timer;

    // Load A and Y (Y is thrown away)
    if (rank == 0) {
        std::cout << "Reading the matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::utility::io::ReadLIBSVM(argv[1], A, Y, skylark::base::ROWS);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";


    /* Compute approximate SVD */
    skylark::nla::approximate_svd_params_t params;
    params.skip_qr = false;
    params.num_iterations = 2;
    int k = 10;

    if (rank == 0) std::cout << "Computing approximate SVD..." << std::endl;
    skylark::nla::ApproximateSVD(A, U, S, V, k, context, params);
    if (rank == 0)
        std::cout <<"Took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    El::Write(U, "out.U", El::ASCII);
    El::Write(S, "out.S", El::ASCII);
    El::Write(V, "out.V", El::ASCII);

    El::Finalize();
    return 0;
}
