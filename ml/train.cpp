#include <elemental.hpp>
#include <skylark.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include "kernels.hpp"
#include "hilbert.hpp"
#include <omp.h>
#include "../base/context.hpp"


namespace bmpi =  boost::mpi;
namespace po = boost::program_options;

typedef skylark::base::sparse_matrix_t<double> sparse_matrix_t;


int main (int argc, char** argv) {
    /* Various MPI/Skylark/Elemental/OpenMP initializations */

	std::cout << "Running skylark_ml" << std::endl;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    bmpi::environment env (argc, argv);
    boost::mpi::communicator comm;

    hilbert_options_t options (argc, argv, comm.size());
    skylark::base::context_t context (options.seed);

    elem::Initialize (argc, argv);

    if (options.exit_on_return) { return -1; }
    if (comm.rank() == 0)
        std::cout << options.print();


    bool sparse = (options.fileformat == LIBSVM_SPARSE) || (options.fileformat == HDF5_SPARSE);
    int flag = 0;

    if (sparse)
        flag = run<sparse_matrix_t, elem::Matrix<double> >(comm, context, options);
    else
        flag = run<LocalMatrixType, LocalMatrixType>(comm, context, options);

    comm.barrier();
    elem::Finalize();
    return flag;
}
