#include <El.hpp>
#include <skylark.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <boost/program_options.hpp>
#include "hilbert.hpp"

#if SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif 

int main(int argc, char* argv[]) {

    El::Initialize (argc, argv);

    boost::mpi::environment env (argc, argv);
    boost::mpi::communicator comm;

    hilbert_options_t options (argc, argv, comm.size());
    skylark::base::context_t context (options.seed);

    if (options.exit_on_return) { return -1; }

    bool sparse = (options.fileformat == LIBSVM_SPARSE)
        || (options.fileformat == HDF5_SPARSE);

    SKYLARK_BEGIN_TRY()

    if (!options.trainfile.empty()) {
        // Training
        if (comm.rank() == 0) {
            std::cout << options.print();
            std::cout << "Mode: Training. Loading data..." << std::endl;
        }

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
    } else if (!options.testfile.empty()) {
        // Testing from file
        if (comm.rank() == 0) {
            std::cout << "Mode: Predicting (from file). Loading data..."
                      << std::endl;
            std::cout << options.print();
        }

        El::Matrix<double> Y;
        El::DistMatrix<double, El::VC, El::STAR> DecisionValues;
        El::DistMatrix<El::Int, El::VC, El::STAR> PredictedLabels;
        El::Int n;
        skylark::ml::hilbert_model_t model(options.modelfile);

        if (sparse) {
            skylark::base::sparse_matrix_t<double> X;
            read(comm, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            boost::mpi::reduce(comm, Y.Height(), n, std::plus<El::Int>(), 0);
            PredictedLabels.Resize(n, 1);
            DecisionValues.Resize(n, model.get_output_size());

            model.predict(X, PredictedLabels.Matrix(), DecisionValues.Matrix(),
                options.numthreads);
        } else {
            El::Matrix<double> X;
            read(comm, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            boost::mpi::reduce(comm, Y.Height(), n, std::plus<El::Int>(), 0);
            PredictedLabels.Resize(n, 1);
            DecisionValues.Resize(n, model.get_output_size());

            model.predict(X, PredictedLabels.Matrix(), DecisionValues.Matrix(),
                options.numthreads);
        }

        double accuracy = 0.0;
        if (model.is_regression()) {

            // TODO can be done better if Y and VC,STAR as well...
            El::Matrix<double> Ye = DecisionValues.Matrix();
            El::Axpy(-1.0, Y, Ye);
            double localerr = std::pow(El::Nrm2(Ye), 2);
            double localnrm = std::pow(El::Nrm2(Y), 2);
            double err, nrm;
            boost::mpi::reduce(comm, localerr, err,
                std::plus<double>(), 0);
            boost::mpi::reduce(comm, localnrm, nrm,
                std::plus<double>(), 0);

            if (comm.rank() == 0)
                    accuracy = std::sqrt(err / nrm);

            if (!options.outputfile.empty())
                El::Write(DecisionValues, options.outputfile, El::ASCII);

        } else {
            El::Int correct = skylark::ml::classification_accuracy(Y,
                DecisionValues.Matrix());
            El::Int totalcorrect, total;
            boost::mpi::reduce(comm, correct, totalcorrect,
                std::plus<El::Int>(), 0);

            if (comm.rank() == 0)
                accuracy =  totalcorrect*100.0/n;

            if (!options.outputfile.empty()) {
                if (options.decisionvals)
                    El::Write(DecisionValues, options.outputfile, El::ASCII);
                else
                    El::Write(PredictedLabels, options.outputfile, El::ASCII);
            }
        }

        if(comm.rank() == 0)
            std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;

        // TODO list?
        // fix logistic case
        // option for probabilities in predicition
        // clean up evaluate
    } else {
        // Preidicting from stdin mode
        skylark::ml::hilbert_model_t model(options.modelfile);

        El::Matrix<double> X, DecisionValues(1, model.get_output_size());
        El::Matrix<El::Int> PredictedLabel(1, 1);

        while(true) {
            std::string line, token;
            std::getline(std::cin, line);
            if (line.empty())
                break;

            El::Zeros(X, model.get_input_size(), 1);

            std::istringstream tokenstream(line);
            while (tokenstream >> token) {
                int delim  = token.find(':');
                std::string ind = token.substr(0, delim);
                std::string val = token.substr(delim+1);
                X.Set(atoi(ind.c_str()) - 1, 0, atof(val.c_str()));
            }

            model.predict(X, PredictedLabel, DecisionValues,
                options.numthreads);

            std::cout <<
                (model.is_regression() ?
                    DecisionValues.Get(0,0) : PredictedLabel.Get(0, 0))
                      << std::endl;
        }
    }

    comm.barrier();

    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT((comm.rank() == 0))

    El::Finalize();

    return 0;
}
