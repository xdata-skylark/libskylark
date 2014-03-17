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
    skylark::sketch::context_t context (12345, world);
    elem::Initialize (argc, argv);
    MPI_Comm mpi_world(world);

    /* Load Commandline options and log them */
    hilbert_options_t options (argc, argv, context.size);
    if (options.exit_on_return) { return -1; }
    if (context.rank==0)
        std::cout << options.print();

    bool sparse = (options.fileformat == LIBSVM_SPARSE);
    int flag = 0;

    if (sparse)
        flag = run<sparse_matrix_t, elem::Matrix<double>>(context, options);
    else
        flag = run<DistInputMatrixType, DistTargetMatrixType>(context, options);

    std::cout << flag << std::endl;
    context.comm.barrier();
    elem::Finalize();
    return flag;
}

/*

    int d, kmax;

    switch(options.fileformat) {
        case LIBSVM_DENSE:
        {
            read_libsvm_dense(context, options.trainfile, X, Y);
            std::cout << " Rank " << context.rank << " on " << env.processor_name() << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << std::endl;
            kmax = *std::max_element(Y.Buffer(), Y.Buffer() + Y.LocalHeight());
            d = X.Height();
            break;
        }
        case LIBSVM_SPARSE:
        {
            read_libsvm_sparse(context, options.trainfile, X, Y);
            std::cout << " Rank " << context.rank << " on " << env.processor_name() << " owns : " << X.Height() <<  " x " << X.Width() << std::endl;
            kmax = *std::max_element(Y.Buffer(), Y.Buffer() + Y.Height());
            d = X.Height();
            break;
        }
        case HDF5:
        {
            #ifdef SKYLARK_HAVE_HDF5
                read_hdf5_dense(context, options.trainfile, X, Y);
                std::cout << " Rank " << context.rank << " on " << env.processor_name() << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << std::endl;
                kmax = *std::max_element(Y.Buffer(), Y.Buffer() + Y.LocalHeight());
                d = X.Height();
            #else
                // TODO
            #endif
            break;
        }
    }



    BlockADMMSolver* Solver = GetSolver(context, options, dimensions);

    // int k = Y.Width();
    int k;


    boost::mpi::all_reduce(context.comm, kmax, k, boost::mpi::maximum<int>());

    if (k>1) // we assume 0-to-N encoding of classes. Hence N = k+1. For two classes, k=1.
        k++;


    elem::Matrix<double> Wbar(features, k);
    elem::MakeZeros(Wbar);

    feature_matrix_t Xv;
    target_matrix_t Yv;


    if (!options.valfile.empty()) {
        context.comm.barrier();

        if(context.rank == 0) std::cout << "Loading validation data." << std::endl;

        switch(options.fileformat) {
            case LIBSVM_DENSE:
            {
                read_libsvm(context, options.valfile, Xv, Yv, d);
                break;

            }
            case LIBSVM_SPARSE:
            {

                read_libsvm(context, options.valfile, Xv, Yv, d);
            }
            #ifdef SKYLARK_HAVE_HDF5
            case HDF5:
            {
                read_hdf5_dense(context, options.valfile, Xv, Yv);
                break;
            #endif
            }
        }

    }


    Solver->train(X, Y, Wbar, Xv, Yv);


    if (context.rank==0) {
        std::stringstream dimensionstring;
        dimensionstring << "# Dimensions " << features << " " << k << "\n";
        elem::Write(Wbar, options.modelfile, elem::ASCII, options.print().append(dimensionstring.str()));
    }

    // Testing - if specified by the user.
    if (!options.testfile.empty()) {
        context.comm.barrier();

        if(context.rank == 0) std::cout << "Starting testing phase." << std::endl;

        feature_matrix_t Xt;
        target_matrix_t Yt;

        switch(options.fileformat) {
        case LIBSVM_DENSE:
            read_libsvm_dense(context, options.testfile, Xt, Yt, d);
            break;
        #ifdef SKYLARK_HAVE_HDF5
        case HDF5:
            read_hdf5_dense(context, options.testfile, Xt, Yt);
            break;
        #endif
        }

        DistTargetMatrixType Yp(Yt.Height(), k);
        Solver->predict(Xt, Yp, Wbar);
        double accuracy = Solver->evaluate(Yt, Yp);

        if(context.rank == 0) std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;
    }


}
*/
