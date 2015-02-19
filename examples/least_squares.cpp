#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skybase = skylark::base;
namespace skysk =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyutil = skylark::utility;
/*******************************************/

const int m = 50000;
const int n = 500;

typedef El::DistMatrix<double, El::VC, El::STAR> matrix_type;
typedef El::DistMatrix<double, El::VC, El::STAR> rhs_type;
typedef El::DistMatrix<double, El::STAR, El::STAR> sol_type;

template<typename MatrixType, typename RhsType, typename SolType>
void check_solution(const MatrixType &A, const RhsType &b, const SolType &x, 
    const RhsType &r0,
    double &res, double &resAtr, double &resFac) {
    RhsType r(b);
    skybase::Gemv(El::NORMAL, -1.0, A, x, 1.0, r);
    res = skybase::Nrm2(r);

    SolType Atr(x.Height(), x.Width(), x.Grid());
    skybase::Gemv(El::TRANSPOSE, 1.0, A, r, 0.0, Atr);
    resAtr = skybase::Nrm2(Atr);

    skybase::Axpy(-1.0, r0, r);
    RhsType dr(b);
    skybase::Axpy(-1.0, r0, dr);
    resFac = skybase::Nrm2(r) / skybase::Nrm2(dr);
}

int main(int argc, char** argv) {
    double res, resAtr, resFac;

    El::Initialize(argc, argv);

    bmpi::communicator world;
    int rank = world.rank();

    skybase::context_t context(23234);

    // Setup problem and righthand side
    // Using Skylark's uniform generator (as opposed to Elemental's)
    // will insure the same A and b are generated regardless of the number
    // of processors.
    matrix_type A =
        skyutil::uniform_matrix_t<matrix_type>::generate(m,
            n, El::DefaultGrid(), context);
    matrix_type b =
        skyutil::uniform_matrix_t<matrix_type>::generate(m,
            1, El::DefaultGrid(), context);

    sol_type x(n,1);
    rhs_type r(b);

    boost::mpi::timer timer;
    double telp;

    // Solve using Elemental. Note: Elemental only supports [MC,MR]...
    El::DistMatrix<double> A1 = A, b1 = b, x1;
    timer.restart();
    El::LeastSquares(El::NORMAL, A1, b1, x1);
    telp = timer.elapsed();
    x = x1;
    check_solution(A, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Elemental:\t\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << "\t\t\t\t\t\t\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;
    double res_opt = res;

    skybase::Gemv(El::NORMAL, -1.0, A, x, 1.0, r);

    // Solve using Sylark
    timer.restart();
    skynla::FastLeastSquares(El::NORMAL, A, b, x, context);
    telp = timer.elapsed();
    check_solution(A, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Skylark:\t\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    // Approximately solve using Sylark
    timer.restart();
    skynla::ApproximateLeastSquares(El::NORMAL, A, b, x, context);
    telp = timer.elapsed();
    check_solution(A, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Skylark (approximate):\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    return 0;
}
