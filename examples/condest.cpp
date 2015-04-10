#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

#include <H5Cpp.h>

namespace bmpi =  boost::mpi;
namespace skybase = skylark::base;
namespace skysketch =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyutil = skylark::utility;

int main(int argc, char** argv) {

    El::Initialize(argc, argv);
    skybase::context_t context(23234);

    skybase::sparse_matrix_t<double> A;
    El::Matrix<double> b;

    boost::mpi::timer timer;

    // Load A and b from HDF5 file
    std::cout << "Reading the matrix... ";
    std::cout.flush();
    timer.restart();
    H5::H5File in(argv[1], H5F_ACC_RDONLY);
    skyutil::io::ReadHDF5(in, "A", A);
    in.close();
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    timer.restart();
    El::Matrix<double> u_min, u_max, v_min, v_max;
    double cond, sigma_min, sigma_min_c, sigma_max;
    skynla::condest_params_t condest_params;
    condest_params.am_i_printing = true;
    condest_params.log_level = 2;
    condest_params.iter_lim = 10000;
    skynla::CondEst(A, cond, sigma_max, v_max, u_max,
        sigma_min, sigma_min_c, v_min, u_min, context, condest_params);
    std::cout <<"Took " << boost::format("%.2e") % timer.elapsed() << " sec\n";
    std::cout << "Condition number = " << cond
              << " sigma_max = " << sigma_max
              << " sigma_min = " << sigma_min << std::endl;

    return 0;
}
