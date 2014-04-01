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


namespace bmpi =  boost::mpi;
namespace po = boost::program_options;
using namespace std;

int main (int argc, char** argv) {
    /* Various MPI/Skylark/Elemental/OpenMP initializations */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    bmpi::environment env (argc, argv);
    bmpi::communicator world;


    hilbert_options_t options (argc, argv, world.size());

    skylark::sketch::context_t context (options.seed, world);
    elem::Initialize (argc, argv);
    MPI_Comm mpi_world(world);

    /* Load Commandline options and log them */

    if (options.exit_on_return) { return -1; }
    if (context.rank==0)
        std::cout << options.print();

    bool sparse = (options.fileformat == LIBSVM_SPARSE) || (options.fileformat == HDF5_SPARSE);
    int flag = 0;

    if (sparse)
        flag = run<sparse_matrix_t, elem::Matrix<double> >(context, options);
    else
        flag = run<LocalMatrixType, LocalMatrixType>(context, options);

    context.comm.barrier();
    elem::Finalize();
    return flag;
}
