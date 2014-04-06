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


int main (int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (argc!=5)
    {
    	std::cout << "convert2hdf5 inputfile hdf5file mode[0:dense,1:sparse] min_d" << std::endl;
    	exit(1);
    }

    std::string inputfile = argv[1];
    std::string hdf5file = argv[2];
    int mode = atoi(argv[3]);
    int min_d  = atoi(argv[4]);

    std::cout << "input: " << inputfile << " hdf5file:" << hdf5file << " mode:" <<  mode << " min_d:" << min_d << std::endl;
    boost::mpi::environment env (argc, argv);

    boost::mpi::communicator world;
    elem::Initialize (argc, argv);

    skylark::sketch::context_t context (12345, world);

    if (mode==0) { // dense
    	LocalMatrixType X;
    	LocalMatrixType Y;
    	read_libsvm(context, inputfile, X, Y, min_d);
    	if(context.rank==0) {
    	        write_hdf5(hdf5file, X,Y);
    		}
    } else {
    	sparse_matrix_t X;
    	LocalMatrixType Y;
    	read_libsvm(context, inputfile, X, Y, min_d);
    	if(context.rank==0) {
    		write_hdf5(hdf5file, X,Y);
    	}
    }

}
