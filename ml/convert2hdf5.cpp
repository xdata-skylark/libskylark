#include <string>
#include <boost/mpi.hpp>
#include <cstdlib>

#define SKYLARK_NO_ANY
#include <skylark.hpp>
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

    boost::mpi::environment env (argc, argv);

    // get communicator
    boost::mpi::communicator comm;
    int rank = comm.rank();

    El::Initialize (argc, argv);


    if (rank == 0)
        std::cout << "input: " << inputfile << " hdf5file:" << hdf5file << " mode:" <<  mode << " min_d:" << min_d << std::endl;

    if (mode==0) { // dense
    	LocalMatrixType X;
    	LocalMatrixType Y;
    	read_libsvm(comm, inputfile, X, Y, min_d);
        write_hdf5(comm, hdf5file, X,Y);
    } else {
    	sparse_matrix_t X;
    	LocalMatrixType Y;
    	read_libsvm(comm, inputfile, X, Y, min_d);
    	if(rank==0)
    		write_hdf5(hdf5file, X,Y);
    }

}
