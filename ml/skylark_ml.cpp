#include <El.hpp>
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


int main(int argc, char* argv[]) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    boost::mpi::environment env (argc, argv);
    boost::mpi::communicator comm;

    hilbert_options_t options (argc, argv, comm.size());
    skylark::base::context_t context (options.seed);

    El::Initialize (argc, argv);

    if (options.exit_on_return) { return -1; }
    if (comm.rank() == 0)
        std::cout << options.print();

    bool sparse = (options.fileformat == LIBSVM_SPARSE)
        || (options.fileformat == HDF5_SPARSE);



    if (!options.trainfile.empty()) {
        // Training
        if (comm.rank() == 0)
            std::cout << "Mode: Training. Loading data..." << std::endl;

        if (sparse) {
            skylark::base::sparse_matrix_t<double> X;
            El::Matrix<double> Y;
            read(comm, options.fileformat, options.trainfile, X, Y);
            skylark::ml::LargeScaleKernelLearning(comm, X, Y, context, options);
        } else {
            El::Matrix<double> X, Y;
            read(comm, options.fileformat, options.trainfile, X, Y);
            skylark::ml::LargeScaleKernelLearning(comm, X, Y, context, options);
        }
    } else {
        // Testing
        if (comm.rank() == 0)
            std::cout << "Mode: Prediciting. Loading data..." << std::endl;

        El::Matrix<double> DecisionValues, Y;
        El::DistMatrix<El::Int, El::VC, El::STAR> PredictedLabels;
        El::Int n;
        skylark::ml::model_t model(options.modelfile);

        if (sparse) {
            skylark::base::sparse_matrix_t<double> X;
            read(comm, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            El::Zeros(DecisionValues, Y.Height(), model.get_output_size());

            boost::mpi::reduce(comm, Y.Height(), n, std::plus<El::Int>(), 0);
            PredictedLabels.Resize(n, 1);

            model.predict(X, PredictedLabels.Matrix(), DecisionValues);
        } else {
            El::Matrix<double> X;
            read(comm, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            El::Zeros(DecisionValues, Y.Height(), model.get_output_size());

            boost::mpi::reduce(comm, Y.Height(), n, std::plus<El::Int>(), 0);
            PredictedLabels.Resize(n, 1);

            model.predict(X, PredictedLabels.Matrix(), DecisionValues);
        }

        El::Int correct = skylark::ml::classification_accuracy(Y, 
            DecisionValues);
        double accuracy = 0.0;
        El::Int totalcorrect, total;
        boost::mpi::reduce(comm, correct, totalcorrect, std::plus<El::Int>(), 0);

        if(comm.rank() == 0) {
            double accuracy =  totalcorrect*100.0/n;
            std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;
        }

        if (!options.outputfile.empty())
            El::Write(PredictedLabels, options.outputfile, El::ASCII);

        // TODO list?
        // fix logistic case
        // option for probabilities in predicition
        // clean up evaluate
    }

    comm.barrier();
    El::Finalize();

    return 0;
}
