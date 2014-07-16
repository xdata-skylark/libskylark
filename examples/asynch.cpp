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
namespace skyutil = skylark::utility;


struct asyprecond_t : 
    public skyalg::outplace_precond_t<elem::Matrix<double>, elem::Matrix<double> > {

    const skybase::sparse_matrix_t<double>& N;
    skybase::context_t &context;

    asyprecond_t(const skybase::sparse_matrix_t<double>& N, 
        skybase::context_t &ctx) 
        : N(N) , context(ctx) { }

    bool is_id() const { return false; }

    void apply(const elem::Matrix<double>& B, elem::Matrix<double>&X) const {
        elem::MakeZeros(X);
        skyalg::AsyRGS(N, B, X, 2, context);
    }

    void apply_adjoint(const elem::Matrix<double>& B, 
        elem::Matrix<double>&X) const {
        // TODO
    }

};

int main(int argc, char** argv) {

    elem::Initialize(argc, argv);
    skybase::context_t context(23234);

    skybase::sparse_matrix_t<double> A;
    elem::Matrix<double> b;

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

    elem::Matrix<double> x(b.Height(), 1);
    std::cout << "Using AsyRGS... ";
    timer.restart();
    elem::MakeZeros(x);
    skyalg::AsyRGS(A, b, x, 20, context);
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    {elem::Matrix<double> r(b);
    skybase::Gemv(elem::TRANSPOSE, -1.0, A, x, 1.0, r);
    double res = skybase::Nrm2(r);
    double nrmb = skybase::Nrm2(b);
    std::cout << "||A x - b||_2 / ||b||_2 = " <<
        boost::format("%.2e") % (res / nrmb) <<
        std::endl;}

    std::cout << "Using CG... ";
    timer.restart();
    elem::MakeZeros(x);
    skyalg::krylov_iter_params_t iter_params;
    iter_params.iter_lim = 20;
    skyalg::CG(A, b, x, iter_params);
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    {elem::Matrix<double> r(b);
    skybase::Gemv(elem::TRANSPOSE, -1.0, A, x, 1.0, r);
    double res = skybase::Nrm2(r);
    double nrmb = skybase::Nrm2(b);
    std::cout << "||A x - b||_2 / ||b||_2 = " <<
        boost::format("%.2e") % (res / nrmb) <<
        std::endl;}

    std::cout << "Using FCG... ";
    timer.restart();
    elem::MakeZeros(x);
    iter_params.iter_lim = 500;
    skyalg::FlexibleCG(A, b, x, iter_params, asyprecond_t(A, context));
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    {elem::Matrix<double> r(b);
    skybase::Gemv(elem::TRANSPOSE, -1.0, A, x, 1.0, r);
    double res = skybase::Nrm2(r);
    double nrmb = skybase::Nrm2(b);
    std::cout << "||A x - b||_2 / ||b||_2 = " <<
        boost::format("%.2e") % (res / nrmb) <<
        std::endl;}


    return 0;
}
