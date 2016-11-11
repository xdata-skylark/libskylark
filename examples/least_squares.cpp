#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

const int m = 50000;
const int n = 500;


template<typename MatrixType, typename RhsType, typename SolType>
void check_solution(const MatrixType &A, const RhsType &b, const SolType &x, 
    const RhsType &r0,
    double &res, double &resAtr, double &resFac) {
    RhsType r(b);
    skylark::base::Gemv(El::NORMAL, -1.0, A, x, 1.0, r);
    res = skylark::base::Nrm2(r);

    SolType Atr(x.Height(), x.Width(), x.Grid());
    skylark::base::Gemv(El::TRANSPOSE, 1.0, A, r, 0.0, Atr);
    resAtr = skylark::base::Nrm2(Atr);

    skylark::base::Axpy(-1.0, r0, r);
    RhsType dr(b);
    skylark::base::Axpy(-1.0, r0, dr);
    resFac = skylark::base::Nrm2(r) / skylark::base::Nrm2(dr);
}

template<typename MatrixType, typename RhsType, typename SolType>
void experiment() {
    typedef MatrixType matrix_type;
    typedef RhsType rhs_type;
    typedef SolType sol_type;

    double res, resAtr, resFac;

    boost::mpi::communicator world;
    int rank = world.rank();

    skylark::base::context_t context(23234);

    // Setup problem and righthand side
    // Using Skylark's uniform generator (as opposed to Elemental's)
    // will insure the same A and b are generated regardless of the number
    // of processors.
    matrix_type A, b;
    skylark::base::UniformMatrix(A, m, n, context);
    skylark::base::UniformMatrix(b, m, 1, context);

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

    // The following computes the optimal residual (r^\star in the logs)
    skylark::base::Gemv(El::NORMAL, -1.0, A, x, 1.0, r);

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF || SKYLARK_HAVE_KISSFFT
    // Solve using Sylark
    timer.restart();
    skylark::nla::FasterLeastSquares(El::NORMAL, A, b, x, context);
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
    skylark::nla::ApproximateLeastSquares(El::NORMAL, A, b, x, context);
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
#else
    std::cout << "You need to have Skylark supporting FFTW or FFTWF " 
              << "to solve with skylark least_squares.cpp"
              << std::endl;
#endif
}



int main(int argc, char** argv) {

    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();

    if (rank == 0)
        std::cout << "Matrix: [VC,STAR], Rhs: [VC,STAR], Sol: [STAR,STAR]\n\n";
    experiment<El::DistMatrix<double, El::VC, El::STAR>,
               El::DistMatrix<double, El::VC, El::STAR>,
               El::DistMatrix<double, El::STAR, El::STAR> > ();

    if (rank == 0)
        std::cout << "\nMatrix: [MC,MR], Rhs: [MC,MR], Sol: [MC,MR]\n\n";
    experiment<El::DistMatrix<double>, El::DistMatrix<double>,
               El::DistMatrix<double> >();

    return 0;
}
