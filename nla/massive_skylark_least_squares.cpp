/*
 * massive_skylark_least_squares.cpp
 *
 *  Created on: Aug 14, 2015
 *      Author: chander
 */
#include <iostream>

#include <El.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>
#include "spi/include/kernel/memory.h"

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


int main(int argc, char** argv) {

    double res, resAtr, resFac;
    double startTime, stopTime ;

    El::Initialize(argc, argv);

    El::mpi::Comm comm = El::mpi::COMM_WORLD;
    const El::Int commRank = El::mpi::Rank( comm );
    const El::Int commSize = El::mpi::Size( comm );

    int numreps = 1 ;
    int samplingfactor = 2 ;

    skylark::base::context_t context(23234);

    if (argc < 2 || argc > 4) {
      std::cerr << "usage: \n" << argv[0] << "\t"
        << "<Matrix Market file name>" << "\t"
        << "[Number of replications]"  << "\t"
        << "[Sampling factor]" << "\t"
        << std::endl;
      return 1;
    }
    std::string mmfilename(argv[1]);

    if (argc >= 3) {
        numreps = atoi(argv[2]) ;
    }

    if (argc >= 4) {
        samplingfactor = atoi(argv[3]) ;
    }

    //Create the Grid object here.
    El::Int gridHeight = El::Grid::FindFactor( commSize );
    El::Grid grid( comm, gridHeight );

    startTime = El::mpi::Time() ;

    //Read the Matrix market file.
    El::DistSparseMatrix<double> ADistSparseMM(comm) ;
    skylark::utility::io::ReadMatrixMarket( ADistSparseMM, mmfilename );

    //Copy into the DistMatrix here.
    El::DistMatrix<double> A(grid), A1(grid) ;
    El::DistMatrix<double> b(grid), b1(grid) ;

    El::Zeros(A, ADistSparseMM.Height(), ADistSparseMM.Width()) ;
    El::Copy( ADistSparseMM, A) ;
    El::Uniform(b, A.Height(), 1);

    if (commRank == 0)
    {
        std::cout << "Before replication" << std::endl ;
        skylark::utility::PrintCNKMeminfo();
    }
    A1.Resize(numreps*A.Height(), A.Width()) ;
    for( int i = 0 ; i < numreps ; i++)
    {
        El::Range<El::Int> rowRange( (i*A.Height()), ((i+1)*A.Height()) );
        El::Range<El::Int> colRange(0, A.Width()) ;
        auto Asub = A1(rowRange, colRange) ;
        Asub = A ;
    }

    // Now make the matrix Dense by adding Gaussian noise N(0,1).
    El::Gaussian(A, A1.Height(), A1.Width());
    El::Axpy(1.0, A, A1) ;

//      skylark::utility::ReplicateDistMatrix(A, numreps) ;
    if(numreps > 1)
        skylark::utility::ReplicateDistMatrix(b, numreps) ;
    if (commRank == 0)
    {
        std::cout << "After replication" << std::endl ;
        skylark::utility::PrintCNKMeminfo();
    }

    El::DistMatrix<double> x(A.Width(), 1, grid);
    El::DistMatrix<double> r(b);

    stopTime  = El::mpi::Time() ;

    if (commRank == 0)
      std::cout << A1.Height() << "\t" << A1.Width() << "\tTotal init time :: " << (stopTime-startTime) << " sec." << std::endl ;

    // Solve using Elemental. Note: Elemental only supports [MC,MR]...
    A = El::LockedView(A1) ;
    b1 = b ;
    El::DistMatrix<double> x1(grid);

    if (commRank == 0)
    {
        std::cout << "Before Least squares" << std::endl ;
        skylark::utility::PrintCNKMeminfo();
    }


    startTime = El::mpi::Time() ;
    El::LeastSquares(El::NORMAL, A1, b1, x1);
    stopTime = El::mpi::Time() ;
    x = x1;
    check_solution(A, b, x, r, res, resAtr, resFac);

    if (commRank == 0)
        std::cout << "Elemental:\t\t\t||r||_2 =  "
              << boost::format("%.4f") % res
              << "\t\t\t\t\t\t\t||A' * r||_2 = " << boost::format("%.4e") % resAtr
              << "\t\tTime: " << (stopTime - startTime) << " sec"
              << std::endl;

    double res_opt = res;

    // The following computes the optimal residual (r^\star in the logs)
    skylark::base::Gemv(El::NORMAL, -1.0, A, x, 1.0, r);

    A1 = A ;
    // Solve using Sylark
    startTime = El::mpi::Time() ;
    skylark::nla::BatchwiseLeastSquares(El::NORMAL, A, A1, b, x, samplingfactor, context);
    stopTime = El::mpi::Time() ;
    check_solution(A, b, x, r, res, resAtr, resFac);
    if (commRank == 0)
        std::cout << "Skylark:\t\t\t||r||_2 =  "
                  << boost::format("%.4f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.4e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.4e") % resAtr
                  << "\t\tTime: " << (stopTime - startTime) << " sec"
                  << std::endl;


    return 0;
}


