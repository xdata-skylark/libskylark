#include <iostream>

#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

#include <H5Cpp.h>

namespace bmpi =  boost::mpi;
namespace skybase = skylark::base;
namespace skysketch =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyml = skylark::ml;
namespace skyutil = skylark::utility;

int main(int argc, char** argv) {

    elem::Initialize(argc, argv);
    skybase::context_t context(23234);

    skybase::sparse_matrix_t<double> A;
    elem::Matrix<double> b;

    boost::mpi::timer timer;

    // Load A and b from HDF5 file
    std::cout << "Reading the adjacency matrix... ";
    std::cout.flush();
    timer.restart();
    H5::H5File in(argv[1], H5F_ACC_RDONLY);
    skyutil::io::ReadHDF5(in, "A", A);
    in.close();
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    // TODO get these as parameters
    std::vector<int> seeds = {19043 - 1};
    double gamma = 5;
    double epsilon = 0.001;
    double alpha = 0.8;

    timer.restart();
    std::vector<int> cluster;
    double cond = skyml::FindLocalCluster(A, seeds, cluster, alpha, gamma, epsilon);
    std::cout <<"Analysis complete! Took "
              << boost::format("%.2e") % timer.elapsed() << " sec\n";
    std::cout << "Cluster found (vertex numbers begin at 1):" << std::endl;
    for (auto it = cluster.begin(); it != cluster.end(); it++)
        std::cout << *it + 1 << " ";
    std::cout << std::endl;
    std::cout << "Conductivity = " << cond << std::endl;

    return 0;
}
