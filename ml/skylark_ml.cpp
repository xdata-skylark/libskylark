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

template<typename XType, typename YType>
void execute_train(const hilbert_options_t &options) {

    // TODO
}

template<>
void execute_train<skylark::base::sparse_matrix_t<double>,
                   El::Matrix<double> > (const hilbert_options_t &options) {

    boost::mpi::communicator world;
    skylark::base::context_t context(options.seed);


    if (options.fileformat == LIBSVM_SPARSE) {

        skylark::base::sparse_star_vr_matrix_t<double> X, Xv;
        El::DistMatrix<double, El::STAR, El::VR> Y, Yv;
        skylark::utility::io::ReadLIBSVM(options.trainfile, X, Y,
            skylark::base::COLUMNS, 0, options.partial);
        if (!options.valfile.empty())
            skylark::utility::io::ReadLIBSVM(options.trainfile, Xv, Yv,
                skylark::base::COLUMNS, skylark::base::Height(X));
        skylark::ml::LargeScaleKernelLearning(world, X.matrix(), Y.Matrix(),
            Xv.matrix(), Yv.Matrix(),context, options);
    } else {

        skylark::base::sparse_matrix_t<double> X, Xv;
        El::Matrix<double> Y, Yv;
        read(world, options.fileformat, options.trainfile, X, Y);
        if (!options.valfile.empty())
            read(world, options.fileformat, options.valfile, Xv, Yv,
                skylark::base::Height(X));
        skylark::ml::LargeScaleKernelLearning(world, X, Y, Xv, Yv,
            context, options);
    }
}

template<>
void execute_train<El::DistMatrix<double, El::STAR, El::VR>,
                   El::DistMatrix<double, El::STAR, El::VR> >
(const hilbert_options_t &options) {

    boost::mpi::communicator world;
    skylark::base::context_t context(options.seed);
    El::DistMatrix<double, El::STAR, El::VR> X, Y, X0, Y0, Xv, Yv;
    int rank = world.rank();

    if(rank == 0)
        std::cout << "Loading training data..." << std::endl;

    switch(options.fileformat) {

    case LIBSVM_DENSE:
        skylark::utility::io::ReadLIBSVM(options.trainfile, X0, Y0,
            skylark::base::COLUMNS, 0, options.partial);
        if (!options.valfile.empty())
            skylark::utility::io::ReadLIBSVM(options.trainfile, Xv, Yv,
                skylark::base::COLUMNS, skylark::base::Height(X0));
        break;

    case HDF5_DENSE: {
        H5::H5File in(options.trainfile, H5F_ACC_RDONLY);
        skylark::utility::io::ReadHDF5(in, "X", X0, -1, options.partial);
        skylark::utility::io::ReadHDF5(in, "Y", Y0, -1, options.partial);
        in.close();

        if (!options.valfile.empty()) {
            if(rank == 0)
                std::cout << "Loading validation data..." << std::endl;

            H5::H5File in(options.valfile, H5F_ACC_RDONLY);
            skylark::utility::io::ReadHDF5(in, "X", Xv);
            skylark::utility::io::ReadHDF5(in, "Y", Yv);
            in.close();
        }
    }
        break;

    default:
        // TODO exception
        break;
    }

    if (options.sample == -1) {

        El::View(X, X0);
        El::View(Y, Y0);

    } else {
        // Sample X and Y
        if (rank == 0) {
            std::cout << "Sampling the data... " << std::endl;
            std::cout.flush();
        }

        skylark::sketch::UST_t<El::DistMatrix<double, El::STAR, El::VR> >
            S(X0.Width(), options.sample, false, context);

        X.Resize(X0.Height(), options.sample);
        S.apply(X0, X, skylark::sketch::rowwise_tag());

        Y.Resize(1, options.sample);
        S.apply(Y0, Y, skylark::sketch::rowwise_tag());
    }

    skylark::ml::LargeScaleKernelLearning(world, X.Matrix(), Y.Matrix(),
        Xv.Matrix(), Yv.Matrix(), context, options);
}

template<typename XType, typename YType>
void execute_predict(const hilbert_options_t &options) {

    // TODO
}

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    boost::mpi::environment env(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();

    hilbert_options_t options(argc, argv, size);

    if (options.exit_on_return) { return -1; }

    bool sparse = (options.fileformat == LIBSVM_SPARSE)
        || (options.fileformat == HDF5_SPARSE);

    SKYLARK_BEGIN_TRY()

    if (!options.trainfile.empty()) {
        // Training
        if (rank == 0) {
            std::cout << options.print();
            std::cout << "Mode: Training." << std::endl;
        }

        if (sparse)
            execute_train<skylark::base::sparse_matrix_t<double>,
                    El::Matrix<double> >(options);
        else
            execute_train<El::DistMatrix<double, El::STAR, El::VR>,
                          El::DistMatrix<double, El::STAR, El::VR> >(options);

    } else if (!options.testfile.empty()) {
        // Testing from file
        if (rank) {
            std::cout << "Mode: Predicting (from file). Loading data..."
                      << std::endl;
            std::cout << options.print();
        }

        El::Matrix<double> Y;
        El::DistMatrix<double, El::VR, El::STAR> DecisionValues;
        El::DistMatrix<El::Int, El::VR, El::STAR> PredictedLabels;
        El::Int n;
        skylark::ml::hilbert_model_t model(options.modelfile);

        if (sparse) {
            skylark::base::sparse_matrix_t<double> X;
            read(world, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            boost::mpi::reduce(world, Y.Height(), n, std::plus<El::Int>(), 0);
            PredictedLabels.Resize(n, 1);
            DecisionValues.Resize(n, model.get_output_size());

            model.predict(X, PredictedLabels.Matrix(), DecisionValues.Matrix(),
                options.numthreads);
        } else {
            El::Matrix<double> X;
            read(world, options.fileformat, options.testfile, X, Y,
                model.get_input_size());

            boost::mpi::reduce(world, Y.Height(), n, std::plus<El::Int>(), 0);
            PredictedLabels.Resize(n, 1);
            DecisionValues.Resize(n, model.get_output_size());

            model.predict(X, PredictedLabels.Matrix(), DecisionValues.Matrix(),
                options.numthreads);
        }

        double accuracy = 0.0;
        if (model.is_regression()) {

            // TODO can be done better if Y is VC,STAR as well...
            El::Matrix<double> Ye = DecisionValues.Matrix();
            El::Axpy(-1.0, Y, Ye);
            double localerr = std::pow(El::Nrm2(Ye), 2);
            double localnrm = std::pow(El::Nrm2(Y), 2);
            double err, nrm;
            boost::mpi::reduce(world, localerr, err,
                std::plus<double>(), 0);
            boost::mpi::reduce(world, localnrm, nrm,
                std::plus<double>(), 0);

            if (rank == 0)
                    accuracy = std::sqrt(err / nrm);

            if (!options.outputfile.empty())
                El::Write(DecisionValues, options.outputfile, El::ASCII);

        } else {
            El::Int correct = skylark::ml::classification_accuracy(Y,
                DecisionValues.Matrix());
            El::Int totalcorrect, total;
            boost::mpi::reduce(world, correct, totalcorrect,
                std::plus<El::Int>(), 0);

            if (rank == 0)
                accuracy =  totalcorrect*100.0/n;

            if (!options.outputfile.empty()) {
                if (options.decisionvals)
                    El::Write(DecisionValues, options.outputfile, El::ASCII);
                else
                    El::Write(PredictedLabels, options.outputfile, El::ASCII);
            }
        }

        if(rank == 0)
            std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;

        // TODO list?
        // fix logistic case
        // option for probabilities in predicition
        // clean up evaluate
    } else {
        // Predicting from stdin mode
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

    world.barrier();

    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT((world.rank() == 0))

    El::Finalize();

    return 0;
}
