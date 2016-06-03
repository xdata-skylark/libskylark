#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>

// Some tricks to make compilation faster
#define SKYLARK_NO_ANY
#define SKYLARK_WITH_GAUSSIAN_RFT_ANY
#define SKYLARK_WITH_LAPLACIAN_RFT_ANY
#define SKYLARK_WITH_PPT_ANY
#define SKYLARK_WITH_FAST_GAUSSIAN_RFT_ANY

#include <skylark.hpp>

// File formats
#define FORMAT_LIBSVM  0
#define FORMAT_HDF5    1

// Algorithms constants
#define CLASSIC_KRR                      0
#define FASTER_KRR                       1
#define APPROXIMATE_KRR                  2
#define SKETCHED_APPROXIMATE_KRR         3
#define FAST_SKETCHED_APPROXIMATE_KRR    4
#define LARGE_SCALE_KRR                  5
#define EXPERIMENTAL_1                 100
#define EXPERIMENTAL_2                 101

// Kernels constants
#define GAUSSIAN_KERNEL   0
#define LAPLACIAN_KERNEL  1
#define POLYNOMIAL_KERNEL 2

#define TRUNCATED_GAUSSIAN_KERNEL   10

std::string cmdline;
int seed = 38734, algorithm = FASTER_KRR, kernel_type = GAUSSIAN_KERNEL;
int fileformat = FORMAT_LIBSVM;
int s = 2000, partial = -1, sketch_size = -1, sample = -1, maxit = 0, maxsplit = 0;
std::string fname, testname, modelname = "model.dat", logfile = "";
double kp1 = 10.0, kp2 = 0.0, kp3 = 1.0, lambda = 0.01, tolerance=0;
bool use_single, use_fast, regression;

#ifndef SKYLARK_AVOID_BOOST_PO

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

int parse_program_options(int argc, char* argv[]) {

    bpo::options_description desc("Options");
    desc.add_options()
        ("help,h", "produce a help message")
        ("trainfile",
            bpo::value<std::string>(&fname),
            "Data to train on (libsvm format).")
        ("testfile",
            bpo::value<std::string>(&testname)->default_value(""),
            "Test data (libsvm format).")
        ("model",
            bpo::value<std::string>(&modelname)->default_value("model.dat"),
            "Name of model file.")
        ("logfile",
            bpo::value<std::string>(&logfile)->default_value(""),
            "File to write log (standard output if empty).")
        ("kernel,k",
             bpo::value<int>(&kernel_type)->default_value(GAUSSIAN_KERNEL),
            "Kernel to use (0: Gaussian, 1: Laplacian, 2: Polynomial).")
        ("algorithm,a",
             bpo::value<int>(&algorithm)->default_value(FASTER_KRR),
            "Algorithm to use (0: Classic, 1: Faster (Precond), "
            "2: Approximate (Random Features)), "
            "3: Sketched Approximate (Piecemeal Random Features + Sketch)), "
            "4: Sketched Approximate w/ Faster Sketching, "
            "5: Large Scale. OPTIONAL.")
        ("seed,s",
            bpo::value<int>(&seed)->default_value(38734),
            "Seed for random number generation. OPTIONAL.")
        ("kernelparam,g",
            bpo::value<double>(&kp1),
            "Kernel parameter. REQUIRED.")
        ("kernelparam2,x",
            bpo::value<double>(&kp2)->default_value(0.0),
            "If Applicable - Second Kernel Parameter (Polynomial Kernel: c).")
        ("kernelparam3,y",
            bpo::value<double>(&kp3)->default_value(1.0),
            "If Applicable - Third Kernel Parameter (Polynomial Kernel: gamma).")
        ("lambda,l",
            bpo::value<double>(&lambda)->default_value(0.01),
            "Lambda regularization parameter.")
        ("tolerance,t",
            bpo::value<double>(&tolerance)->default_value(0),
            "Tolerance for the iterative method (when used). "
            "0 will default based on algorithm (1e-3 for -a 1, 1e-1 for -a 5)")
        ("maxsplit,c",
            bpo::value<int>(&maxsplit)->default_value(0),
            "Maximum number of random features in a split for large scale "
            "algorithms (-a 3 to 5). 0 will default to 2 * input dimension. ")
        ("maxit,i",
            bpo::value<int>(&maxit)->default_value(0),
            "Maximum number of iterations for the iterative method (when used). "
            "0 will default based on algorithm (1000 for -a 1, 20 for -a 5)")
        ("partial,p",
            bpo::value<int>(&partial)->default_value(-1),
            "Load only specified quantity examples from training. "
            "Will read all if -1.")
        ("sample,z",
            bpo::value<int>(&sample)->default_value(-1),
            "Sample the input data. Will use all if -1. ")
        ("single", "Whether to use single precision instead of double.")
        ("fast", "Try using a fast feature transform.")
        ("regression", "Build a regression model"
            "(default is classification).")
        ("numfeatures,f",
            bpo::value<int>(&s),
            "Number of random features (if relevant).")
        ("sketchsize,r",
            bpo::value<int>(&sketch_size)->default_value(-1),
            "Sketch size (for regression problem; if relevant (i.e., -a 3). "
            "-1 - will be determined by software. ")
        ("fileformat",
            po::value<int>(&fileformat)->default_value(FORMAT_LIBSVM),
            "Fileformat (default: 0 (libsvm), 1 (hdf5)");

    bpo::positional_options_description positional;
    positional.add("trainfile", 1);
    positional.add("testfile", 2);

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0]
                      << " [options] input-file-name [test-file-name]"
                      << std::endl;
            std::cout << desc;
            return 0;
        }

        if (!vm.count("trainfile")) {
            std::cout << "Input file is required." << std::endl;
            return -1;
        }

        bpo::notify(vm);

        use_single = vm.count("single");
        use_fast = vm.count("fast");
        regression = vm.count("regression");

    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    return 1000;
}


#else

int parse_program_options(int argc, char* argv[]) {

    int poscount = 0;
    for (int i = 1; i < argc; i += 2) {
        std::string flag = argv[i];
        std::string value = i + 1 < argc ? argv[i+1] : "";

        if (flag == "--seed" || flag == "-s")
            seed = boost::lexical_cast<int>(value);

        if (flag == "--lambda" || flag == "-l")
            lambda = boost::lexical_cast<double>(value);

        if (flag == "--tolerance" || flag == "-t")
            tolerance = boost::lexical_cast<double>(value);

        if (flag == "--maxit" || flag == "-i")
            maxit = boost::lexical_cast<int>(value);

        if (flag == "--maxsplit" || flag == "-c")
            maxsplit = boost::lexical_cast<int>(value);

        if (flag == "--partial" || flag == "-p")
            partial = boost::lexical_cast<int>(value);

        if (flag == "--sample" || flag == "-z")
            sample = boost::lexical_cast<int>(value);

        if (flag == "--kernelparam" || flag == "-g")
            kp1 = boost::lexical_cast<double>(value);

        if (flag == "--kernelparam2" || flag == "-x")
            kp2 = boost::lexical_cast<double>(value);

        if (flag == "--kernelparam3" || flag == "-y")
            kp3 = boost::lexical_cast<double>(value);

        if (flag == "--kernel" || flag == "-k")
            kernel_type = boost::lexical_cast<int>(value);

        if (flag == "--algorithm" || flag == "-a")
            algorithm = boost::lexical_cast<int>(value);

        if (flag == "--fileformat")
            fileformat = boost::lexical_cast<int>(value);

        if (flag == "--nunmfeatures" || flag == "-f")
            s = boost::lexical_cast<int>(value);

        if (flag == "--sketchsize" || flag == "-r")
            sketch_size = boost::lexical_cast<int>(value);

        if (flag == "--single") {
            use_single = true;
            i--;
        }

        if (flag == "--regression") {
            regression = true;
            i--;
        }

        if (flag == "--fast") {
            use_fast = true;
            i--;
        }

        if (flag == "--trainfile")
            fname = value;

        if (flag == "--logfile")
            logfile = value;

        if (flag == "--model")
            modelname = value;

        if (flag == "--testfile")
            testname = value;

        if (flag[0] != '-' && poscount != 0)
            testname = flag;

        if (flag[0] != '-' && poscount == 0) {
            fname = flag;
            poscount++;
        }

        if (flag[0] != '-')
            i--;
    }

    return 1000;
}

#endif

template<typename T>
int execute_classification(skylark::base::context_t &context) {

    boost::mpi::communicator world;
    int rank = world.rank();

    std::ostream *log_stream = &std::cout;
    if (rank == 0 && logfile != "") {
        log_stream = new std::ofstream();
        ((std::ofstream *)log_stream)->open(logfile);
    }

    El::DistMatrix<T> X0, X;
    El::DistMatrix<El::Int> L0, L;

    boost::mpi::timer timer;

    if (rank == 0) {
        *log_stream << "# Generated using kernel_regression ";
        *log_stream << "using the following command-line: " << std::endl;
        *log_stream << "#\t" << cmdline << std::endl;
        *log_stream << "# Number of ranks is " << world.size() << std::endl;
    }

    // Load X and L
    if (rank == 0) {
        *log_stream << "Reading the matrix... ";
        log_stream->flush();
        timer.restart();
    }

    switch (fileformat) {
    case FORMAT_LIBSVM:
        skylark::utility::io::ReadLIBSVM(fname, X0, L0, skylark::base::COLUMNS,
            0, partial);
        break;

#ifdef SKYLARK_HAVE_HDF5
    case FORMAT_HDF5: {
        H5::H5File in(fname, H5F_ACC_RDONLY);
        skylark::utility::io::ReadHDF5(in, "X", X0, -1, partial);
        skylark::utility::io::ReadHDF5(in, "Y", L0, -1, partial);
        in.close();
    }
        break;
#endif

    default:
        *log_stream << "Invalid file format specified." << std::endl;
        return -1;
    }

    if (rank == 0)
        *log_stream <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (sample == -1) {

        El::View(X, X0);
        El::View(L, L0);

    } else {
        // Sample X and L
        if (rank == 0) {
            *log_stream << "Sampling the data... ";
            log_stream->flush();
            timer.restart();
        }


        skylark::sketch::UST_data_t S(X0.Width(), sample, false, context);

        X.Resize(X0.Height(), sample);
        skylark::sketch::UST_t<El::DistMatrix<T> > (S).apply(X0,
            X, skylark::sketch::rowwise_tag());

        L.Resize(1, sample);
        skylark::sketch::UST_t<El::DistMatrix<El::Int> >(S).apply(L0,
            L, skylark::sketch::rowwise_tag());

        if (rank == 0)
            *log_stream <<"took " << boost::format("%.2e") % timer.elapsed()
                        << " sec\n";
    }

    // Training
    if (rank == 0) {
        *log_stream << "Training... " << std::endl;
        timer.restart();
    }

    std::shared_ptr<skylark::ml::kernel_t> k_ptr;

    switch (kernel_type) {
    case GAUSSIAN_KERNEL:
        k_ptr.reset(new skylark::ml::gaussian_t(X.Height(), kp1));
        break;

    case LAPLACIAN_KERNEL:
        k_ptr.reset(new skylark::ml::laplacian_t(X.Height(), kp1));
        break;

    case POLYNOMIAL_KERNEL:
        k_ptr.reset(new skylark::ml::polynomial_t(X.Height(), kp1, kp2, kp3));
        break;

    case TRUNCATED_GAUSSIAN_KERNEL:
        k_ptr.reset(new skylark::ml::truncated_gaussian_t(X.Height(), kp1, kp2));
        break;

    default:
        *log_stream << "Invalid kernel specified." << std::endl;
        return -1;
    }

    skylark::ml::kernel_container_t k(k_ptr);

    El::DistMatrix<T> A, W;
    std::vector<El::Int> rcoding;

    skylark::sketch::sketch_transform_container_t<El::DistMatrix<T>,
                                                  El::DistMatrix<T> >  S;
    bool scale_maps = true;
    std::vector<
        skylark::sketch::sketch_transform_container_t<El::DistMatrix<T>,
                                                      El::DistMatrix<T> > > transforms;

    skylark::ml::rlsc_params_t rlsc_params(rank == 0, 4, *log_stream, "\t");
    rlsc_params.use_fast = use_fast;

    skylark::ml::model_t<El::Int, T> *model;

    switch(algorithm) {
    case CLASSIC_KRR:
        skylark::ml::KernelRLSC(skylark::base::COLUMNS, k, X, L,
            T(lambda), A, rcoding, rlsc_params);
        model =
            new skylark::ml::kernel_model_t<skylark::ml::kernel_container_t,
                  El::Int, T>(k, skylark::base::COLUMNS, X, fname, fileformat,
                      A, rcoding);
        break;

    case FASTER_KRR:
        rlsc_params.iter_lim = (maxit == 0) ? 1000 : maxit;
        rlsc_params.tolerance = (tolerance == 0) ? 1e-3 : tolerance;
        skylark::ml::FasterKernelRLSC(skylark::base::COLUMNS, k, X, L,
            T(lambda), A, rcoding, s, context, rlsc_params);
        model =
            new skylark::ml::kernel_model_t<skylark::ml::kernel_container_t,
                  El::Int, T>(k, skylark::base::COLUMNS, X, fname, fileformat,
                      A, rcoding);
        break;

    case APPROXIMATE_KRR:
        skylark::ml::ApproximateKernelRLSC(skylark::base::COLUMNS, k, X, L,
            T(lambda), S, W, rcoding, s, context, rlsc_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, El::Int, T>
            (S, W, rcoding);
        break;

    case SKETCHED_APPROXIMATE_KRR:
    case FAST_SKETCHED_APPROXIMATE_KRR:
        rlsc_params.sketched_rls = true;
        rlsc_params.sketch_size = sketch_size;
        rlsc_params.fast_sketch = algorithm == FAST_SKETCHED_APPROXIMATE_KRR;
        rlsc_params.max_split = maxsplit;
        skylark::ml::SketchedApproximateKernelRLSC(skylark::base::COLUMNS, k, X, L,
            T(lambda), scale_maps, transforms, W, rcoding, s, sketch_size,
            context, rlsc_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, El::Int, T>
            (scale_maps, transforms, W, rcoding);
        break;

    case LARGE_SCALE_KRR:
        rlsc_params.iter_lim = (maxit == 0) ? 20 : maxit;
        rlsc_params.tolerance = (tolerance == 0) ? 1e-1 : tolerance;
        rlsc_params.max_split = maxsplit;
        skylark::ml::LargeScaleKernelRLSC(skylark::base::COLUMNS, k, X, L,
            T(lambda), scale_maps, transforms, W, rcoding, s,
            context, rlsc_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, El::Int, T>
            (scale_maps, transforms, W, rcoding);
        break;

    case EXPERIMENTAL_1:
    case EXPERIMENTAL_2:
        rlsc_params.sketched_rls = true;
        rlsc_params.fast_sketch = algorithm == EXPERIMENTAL_2;
        skylark::ml::ApproximateKernelRLSC(skylark::base::COLUMNS, k, X, L,
            T(lambda), S, W, rcoding, s, context, rlsc_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, El::Int, T>
            (S, W, rcoding);
        break;

    default:
        *log_stream << "Invalid algorithm value specified." << std::endl;
        return -1;
    }

    if (rank == 0)
        *log_stream << "Training took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (modelname != "NOSAVE") {
        // Save model
        if (rank == 0) {
            *log_stream << "Saving model... ";
            log_stream->flush();
            timer.restart();
        }

        boost::property_tree::ptree pt = model->to_ptree();

        if (rank == 0) {
            std::ofstream of(modelname);
            of << "# Generated using kernel_regression ";
            of << "using the following command-line: " << std::endl;
            of << "#\t" << cmdline << std::endl;
            of << "# Number of ranks is " << world.size() << std::endl;
            boost::property_tree::write_json(of, pt);
            of.close();
        }

        if (rank == 0)
            *log_stream <<"took " << boost::format("%.2e") % timer.elapsed()
                        << " sec\n";
    }

    // Test
    if (!testname.empty()) {
        if (rank == 0) {
            *log_stream << "Predicting... ";
            log_stream->flush();
            timer.restart();
        }

        El::DistMatrix<T> XT;
        El::DistMatrix<El::Int> LT;

        switch (fileformat) {
        case FORMAT_LIBSVM:
            skylark::utility::io::ReadLIBSVM(testname, XT, LT,
                skylark::base::COLUMNS, X.Height());
            break;

#ifdef SKYLARK_HAVE_HDF5
        case FORMAT_HDF5: {
            H5::H5File in(testname, H5F_ACC_RDONLY);
            skylark::utility::io::ReadHDF5(in, "X", XT);
            skylark::utility::io::ReadHDF5(in, "Y", LT);
            in.close();
        }
            break;
#endif

        default:
            *log_stream << "Invalid file format specified." << std::endl;
            return -1;
        }

        El::DistMatrix<El::Int> LP;
        model->predict(skylark::base::COLUMNS, XT, LP);

        if (rank == 0)
            *log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                      << " sec\n";

        int errs = 0;
        if (LT.LocalHeight() > 0)
            for(int i = 0; i < LT.LocalWidth(); i++)
                if (LT.GetLocal(0, i) != LP.GetLocal(0, i))
                    errs++;

        errs = El::mpi::AllReduce(errs, MPI_SUM, LT.DistComm());

        if (rank == 0)
            *log_stream << "Error rate: "
                      << boost::format("%.2f") % ((errs * 100.0) / LT.Width())
                      << "%" << std::endl;
    }

    if (rank == 0 && logfile != "") {
        ((std::ofstream *)log_stream)->close();
        delete log_stream;
    }

    delete model;

    return 0;
}

template<typename T>
int execute_regression(skylark::base::context_t &context) {

    boost::mpi::communicator world;
    int rank = world.rank();

    std::ostream *log_stream = &std::cout;
    if (rank == 0 && logfile != "") {
        log_stream = new std::ofstream();
        ((std::ofstream *)log_stream)->open(logfile);
    }

    El::DistMatrix<T> X0, X, Y0, Y;

    boost::mpi::timer timer;

    if (rank == 0) {
        *log_stream << "# Generated using kernel_regression ";
        *log_stream << "using the following command-line: " << std::endl;
        *log_stream << "#\t" << cmdline << std::endl;
        *log_stream << "# Number of ranks is " << world.size() << std::endl;
    }

    // Load X and Y
    if (rank == 0) {
        *log_stream << "Reading the matrix... ";
        log_stream->flush();
        timer.restart();
    }

    switch (fileformat) {
    case FORMAT_LIBSVM:
        skylark::utility::io::ReadLIBSVM(fname, X0, Y0, skylark::base::COLUMNS,
            0, partial);
        break;

#ifdef SKYLARK_HAVE_HDF5
    case FORMAT_HDF5: {
        H5::H5File in(fname, H5F_ACC_RDONLY);
        skylark::utility::io::ReadHDF5(in, "X", X0, -1, partial);
        skylark::utility::io::ReadHDF5(in, "Y", Y0, -1, partial);
        in.close();
    }
        break;
#endif

    default:
        *log_stream << "Invalid file format specified." << std::endl;
        return -1;
    }

    if (rank == 0)
        *log_stream <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (sample == -1) {

        El::View(X, X0);
        El::View(Y, Y0);

    } else {
        // Sample X and Y
        if (rank == 0) {
            *log_stream << "Sampling the data... ";
            log_stream->flush();
            timer.restart();
        }


        skylark::sketch::UST_data_t S(X0.Width(), sample, false, context);

        X.Resize(X0.Height(), sample);
        skylark::sketch::UST_t<El::DistMatrix<T> > (S).apply(X0,
            X, skylark::sketch::rowwise_tag());

        Y.Resize(1, sample);
        skylark::sketch::UST_t<El::DistMatrix<T> > (S).apply(Y0,
            Y, skylark::sketch::rowwise_tag());

        if (rank == 0)
            *log_stream <<"took " << boost::format("%.2e") % timer.elapsed()
                        << " sec\n";
    }

    // Training
    if (rank == 0) {
        *log_stream << "Training... " << std::endl;
        timer.restart();
    }

    std::shared_ptr<skylark::ml::kernel_t> k_ptr;

    switch (kernel_type) {
    case GAUSSIAN_KERNEL:
        k_ptr.reset(new skylark::ml::gaussian_t(X.Height(), kp1));
        break;

    case LAPLACIAN_KERNEL:
        k_ptr.reset(new skylark::ml::laplacian_t(X.Height(), kp1));
        break;

    case POLYNOMIAL_KERNEL:
        k_ptr.reset(new skylark::ml::polynomial_t(X.Height(), kp1, kp2, kp3));
        break;

    case TRUNCATED_GAUSSIAN_KERNEL:
        k_ptr.reset(new skylark::ml::truncated_gaussian_t(X.Height(), kp1, kp2));
        break;

    default:
        *log_stream << "Invalid kernel specified." << std::endl;
        return -1;
    }

    skylark::ml::kernel_container_t k(k_ptr);

    El::DistMatrix<T> A, W;

    skylark::sketch::sketch_transform_container_t<El::DistMatrix<T>,
                                                  El::DistMatrix<T> >  S;
    bool scale_maps = true;
    std::vector<
        skylark::sketch::sketch_transform_container_t<El::DistMatrix<T>,
                                                      El::DistMatrix<T> > > transforms;

    skylark::ml::krr_params_t krr_params(rank == 0, 4, *log_stream, "\t");
    krr_params.use_fast = use_fast;

    skylark::ml::model_t<T, T> *model;

    // Transpose Y since KernelRidge expects it to be a column vector (TODO ?)
    El::DistMatrix<T> Ytransp;
    El::Transpose(Y, Ytransp, true);
    
    switch(algorithm) {
    case CLASSIC_KRR:
        skylark::ml::KernelRidge(skylark::base::COLUMNS, k, X, Ytransp,
            T(lambda), A, krr_params);
        model =
            new skylark::ml::kernel_model_t<skylark::ml::kernel_container_t,
                  T, T>(k, skylark::base::COLUMNS, X, fname, fileformat,
                      A);
        break;

    case FASTER_KRR:
        krr_params.iter_lim = (maxit == 0) ? 1000 : maxit;
        krr_params.tolerance = (tolerance == 0) ? 1e-3 : tolerance;
        skylark::ml::FasterKernelRidge(skylark::base::COLUMNS, k, X, Ytransp,
            T(lambda), A, s, context, krr_params);
        model =
            new skylark::ml::kernel_model_t<skylark::ml::kernel_container_t,
                  T, T>(k, skylark::base::COLUMNS, X, fname, fileformat, A);
        break;

    case APPROXIMATE_KRR:
        skylark::ml::ApproximateKernelRidge(skylark::base::COLUMNS, k, X, Ytransp,
            T(lambda), S, W, s, context, krr_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, T, T>
            (S, W);
        break;

    case SKETCHED_APPROXIMATE_KRR:
    case FAST_SKETCHED_APPROXIMATE_KRR:
        krr_params.sketched_rr = true;
        krr_params.sketch_size = sketch_size;
        krr_params.fast_sketch = algorithm == FAST_SKETCHED_APPROXIMATE_KRR;
        krr_params.max_split = maxsplit;
        skylark::ml::SketchedApproximateKernelRidge(skylark::base::COLUMNS, k,
            X, Ytransp,
            T(lambda), scale_maps, transforms, W, s, sketch_size,
            context, krr_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, T, T>
            (scale_maps, transforms, W);
        break;

    case LARGE_SCALE_KRR:
        krr_params.iter_lim = (maxit == 0) ? 20 : maxit;
        krr_params.tolerance = (tolerance == 0) ? 1e-1 : tolerance;
        krr_params.max_split = maxsplit;
        skylark::ml::LargeScaleKernelRidge(skylark::base::COLUMNS, k, X, Ytransp,
            T(lambda), scale_maps, transforms, W, s,
            context, krr_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, T, T>
            (scale_maps, transforms, W);
        break;

    case EXPERIMENTAL_1:
    case EXPERIMENTAL_2:
        krr_params.sketched_rr = true;
        krr_params.fast_sketch = algorithm == EXPERIMENTAL_2;
        skylark::ml::ApproximateKernelRidge(skylark::base::COLUMNS, k, X, Ytransp,
            T(lambda), S, W, s, context, krr_params);
        model =
            new skylark::ml::feature_expansion_model_t<
                skylark::sketch::sketch_transform_container_t, T, T>
            (S, W);
        break;

    default:
        *log_stream << "Invalid algorithm value specified." << std::endl;
        return -1;
    }

    if (rank == 0)
        *log_stream << "Training took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (modelname != "NOSAVE") {
        // Save model
        if (rank == 0) {
            *log_stream << "Saving model... ";
            log_stream->flush();
            timer.restart();
        }

        boost::property_tree::ptree pt = model->to_ptree();

        if (rank == 0) {
            std::ofstream of(modelname);
            of << "# Generated using kernel_regression ";
            of << "using the following command-line: " << std::endl;
            of << "#\t" << cmdline << std::endl;
            of << "# Number of ranks is " << world.size() << std::endl;
            boost::property_tree::write_json(of, pt);
            of.close();
        }

        if (rank == 0)
            *log_stream <<"took " << boost::format("%.2e") % timer.elapsed()
                        << " sec\n";
    }

    // Test
    if (!testname.empty()) {
        if (rank == 0) {
            *log_stream << "Predicting... ";
            log_stream->flush();
            timer.restart();
        }

        El::DistMatrix<T> XT;
        El::DistMatrix<T> YT;

        switch (fileformat) {
        case FORMAT_LIBSVM:
            skylark::utility::io::ReadLIBSVM(testname, XT, YT,
                skylark::base::COLUMNS, X.Height());
            break;

#ifdef SKYLARK_HAVE_HDF5
        case FORMAT_HDF5: {
            H5::H5File in(testname, H5F_ACC_RDONLY);
            skylark::utility::io::ReadHDF5(in, "X", XT);
            skylark::utility::io::ReadHDF5(in, "Y", YT);
            in.close();
        }
            break;
#endif

        default:
            *log_stream << "Invalid file format specified." << std::endl;
            return -1;
        }

        El::DistMatrix<T> YP;
        model->predict(skylark::base::COLUMNS, XT, YP);

        if (rank == 0)
            *log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                      << " sec\n";

        T nrm_Yt = El::Nrm2(YT);
        El::Axpy(T(-1.0), YP, YT);
        T nrm_E = El::Nrm2(YT);

        if (rank == 0)
            *log_stream << "Error rate: "
                        << boost::format("%.4e") % (nrm_E / nrm_Yt)
                        << std::endl;
        world.barrier();
    }

    if (rank == 0 && logfile != "") {
        ((std::ofstream *)log_stream)->close();
        delete log_stream;
    }

    delete model;

    return 0;
}

int main(int argc, char* argv[]) {

    for(int i = 0; i < argc; i++) {
        cmdline.append(argv[i]);
        if (i < argc - 1)
            cmdline.append(" ");
    }

    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();

    int flag = parse_program_options(argc, argv);

    if (flag != 1000)
        return flag;

    skylark::base::context_t context(seed);

    int ret = -1;

    SKYLARK_BEGIN_TRY()

        if (regression) {
            if (use_single)
                ret = execute_regression<float>(context);
            else
                ret = execute_regression<double>(context);
        } else {
            if (use_single)
                ret = execute_classification<float>(context);
            else
                ret = execute_classification<double>(context);
        }
    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT((rank == 0))

        catch (const std::exception& ex) {
            if (rank == 0) SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
        }

    El::Finalize();

    return ret;
}
