#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
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
    std::cout << "Reading the matrix and rhs... ";
    std::cout.flush();
    timer.restart();
    H5::H5File in(argv[1], H5F_ACC_RDONLY);
    skyutil::io::ReadHDF5(in, "A", A);
    skyutil::io::ReadHDF5(in, "b", b);
    in.close();
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    El::Matrix<double> x(b.Height(), 1);
    std::cout << "Using AsyRGS... " << std::endl;
    timer.restart();
    El::Zero(x);
    skyalg::asy_iter_params_t asy_params;
    asy_params.tolerance = 1e-3;
    asy_params.syn_sweeps = 5;
    asy_params.sweeps_lim = 2000;
    asy_params.am_i_printing = true;
    asy_params.log_level = 2;
    skyalg::AsyRGS(A, b, x, context, asy_params);
    std::cout <<"Took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    std::cout << "Using CG... " << std::endl;
    timer.restart();
    El::Zero(x);
    skyalg::krylov_iter_params_t krylov_params;
    krylov_params.tolerance = 1e-3;
    krylov_params.res_print = 30;
    krylov_params.iter_lim = 2000;
    krylov_params.am_i_printing = true;
    krylov_params.log_level = 2;
    skyalg::CG(A, b, x, krylov_params);
    std::cout <<"Took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    std::cout << "Using FCG (high accuracy)... " << std::endl;
    timer.restart();
    El::Zero(x);
    asy_params.tolerance = 1e-8;
    asy_params.sweeps_lim = 2;
    asy_params.syn_sweeps = 0;
    asy_params.iter_lim = 200;
    asy_params.iter_res_print = 20;
    asy_params.am_i_printing = true;
    asy_params.log_level = 2;
    skyalg::AsyFCG(A, b, x, context, asy_params);
    std::cout <<"Took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    return 0;
}
