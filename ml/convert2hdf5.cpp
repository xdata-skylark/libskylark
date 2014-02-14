/*
 * convert2hdf5.cpp
 *
 *  Created on: Feb 13, 2014
 *      Author: vikas
 */


#include <string>
#include <skylark.hpp>
#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <cstdlib>
#include "io.hpp"

typedef elem::DistMatrix<double, elem::STAR, elem::VC> DistInputMatrixType;

// Rows are examples, columns are target values
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistTargetMatrixType;


int main (int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    std::string inputfile = argv[1];
    std::string hdf5file = argv[2];
    int min_d  = atoi(argv[3]);

    std::cout << inputfile << " " << hdf5file << std::endl;
    boost::mpi::environment env (argc, argv);

    boost::mpi::communicator world;
    elem::Initialize (argc, argv);

    skylark::sketch::context_t context (12345, world);

    DistInputMatrixType X;
    DistTargetMatrixType Y;

    std::cout << inputfile << " " << hdf5file << std::endl;

    read_libsvm_dense(context, inputfile, X, Y, min_d);

    elem::Matrix<double> x = X.Matrix();
    elem::Matrix<double> y = Y.Matrix();

    if(context.rank==0) {
        write_elem_hdf5(hdf5file, x,y);
    }

}
