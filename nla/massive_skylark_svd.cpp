/*
 * massive_skylark_svd.cpp
 *
 *  Created on: Aug 7, 2015
 *      Author: chander
 */

#ifndef MASSIVE_SKYLARK_SVD_CPP_
#define MASSIVE_SKYLARK_SVD_CPP_

#include <iostream>
#include <El.hpp>
#include <skylark.hpp>


int main(int argc, char** argv) {

    double res, resAtr, resFac;
    double startTime, stopTime ;

    El::Initialize(argc, argv);

    El::mpi::Comm comm = El::mpi::COMM_WORLD;
    const El::Int commRank = El::mpi::Rank( comm );
    const El::Int commSize = El::mpi::Size( comm );

    int topk = 100 ;
    int oversampling_ratio = 10, oversampling_additive = 0, seed = 38734 ;

    skylark::base::context_t context(23234);

    if (argc < 2 || argc > 4) {
      std::cerr << "usage: \n" << argv[0] << "\t"
        << "<Matrix Market file name>" << "\t"
        << "[top-k]" << "\t"
        << "[Oversampling ratio for top-k]"
        << std::endl;
      return 1;
    }
    std::string mmfilename(argv[1]);

    if (argc >= 3) {
    	topk = atoi(argv[2]) ;
    }

    if (argc >= 4) {
    	oversampling_ratio = atoi(argv[3]) ;
    }

    //Create the Grid object here.
    El::Int gridHeight = El::Grid::FindFactor( commSize );
    El::Grid grid( comm, gridHeight );

    startTime = El::mpi::Time() ;

    //Read the Matrix market file.
    El::DistSparseMatrix<double> ADistSparseMM(comm) ;
    skylark::utility::io::ReadMatrixMarket( ADistSparseMM, mmfilename );

    //Copy into the DistMatrix here.
    El::DistMatrix<double> A(grid), A1(grid), U(grid), V(grid) ;
    El::DistMatrix<double, El::VR, El::STAR> S(grid) ;

    El::Zeros(A, ADistSparseMM.Height(), ADistSparseMM.Width()) ;
    El::Copy( ADistSparseMM, A) ;


    if (commRank == 0)
    {
        std::cout << "Before replication" << std::endl ;
        skylark::utility::PrintCNKMeminfo();
    }
    // Now make the matrix Dense by adding Gaussian noise N(0,1).
    El::Gaussian(A1, A.Height(), A.Width());
    El::Axpy(1.0, A1, A) ;

	if (commRank == 0)
	{
		std::cout << "After replication" << std::endl ;
		skylark::utility::PrintCNKMeminfo();
	}
	stopTime  = El::mpi::Time() ;

	if (commRank == 0)
	  std::cout << A.Height() << "\t" << A.Width() << "\tTotal init time :: " << (stopTime-startTime) << " sec." << std::endl ;

	skylark::nla::massive_svd_params_t params ;
    params.oversampling_ratio = oversampling_ratio;
    params.oversampling_additive = oversampling_additive;

    startTime = El::mpi::Time() ;
    skylark::nla::MassiveSVD(A, U, S, V, topk, context, params) ;
    stopTime  = El::mpi::Time() ;
    if (commRank == 0)
        std::cout << A.Height() << "\t" << A.Width() << "\tMassive SVD time :: " << (stopTime-startTime) << " sec." << std::endl ;


}


#endif /* MASSIVE_SKYLARK_SVD_CPP_ */
