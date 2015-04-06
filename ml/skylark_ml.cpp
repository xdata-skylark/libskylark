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

        El::Matrix<double> DecisionValues, PredictedLabels, Y;
        // TODO model shouldn't depend on input-output types.
        if (sparse) {
            skylark::ml::model_t<skylark::base::sparse_matrix_t<double>,
                                 El::Matrix<double> > model(options.modelfile);

            skylark::base::sparse_matrix_t<double> X;
            read(comm, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            DecisionValues.Resize(Y.Height(), model.get_output_size());
            PredictedLabels.Resize(Y.Height(), 1);

            El::Zero(DecisionValues);
            El::Zero(PredictedLabels);

            model.predict(X, PredictedLabels, DecisionValues);
        } else {
            skylark::ml::model_t<El::Matrix<double>,
                                 El::Matrix<double> > model(options.modelfile);

            El::Matrix<double> X;
            read(comm, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            DecisionValues.Resize(Y.Height(), model.get_output_size());
            PredictedLabels.Resize(Y.Height(), 1);

            El::Zero(DecisionValues);
            El::Zero(PredictedLabels);

            model.predict(X, PredictedLabels, DecisionValues);
        }

        El::Int correct = skylark::ml::classification_accuracy(Y, 
            DecisionValues);
        double accuracy = 0.0;
        El::Int totalcorrect, total;
        boost::mpi::reduce(comm, correct, totalcorrect, std::plus<El::Int>(), 0);
        boost::mpi::reduce(comm, Y.Height(), total, std::plus<El::Int>(), 0);

        if(comm.rank() == 0) {
            double accuracy =  totalcorrect*100.0/total;
            std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;
        }

        // TODO list?
        // fix logistic case
        // provide mechanism to dump predictions
        // clean up evaluate
    }

    comm.barrier();
    El::Finalize();

    return 0;
}
