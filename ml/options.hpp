#ifndef SKYLARK_HILBERT_OPTIONS_HPP
#define SKYLARK_HILBERT_OPTIONS_HPP

#ifndef SKYLARK_AVOID_BOOST_PO

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#else

#include <boost/lexical_cast.hpp>

#endif

#define DEFAULT_LAMBDA 0.0
#define DEFAULT_RHO 1.0
#define DEFAULT_THREADS 1
#define DEFAULT_FEATURE_PARTITIONS 1
#define DEFAULT_KERPARAM 1.0
#define DEFAULT_TOL 0.001
#define DEFAULT_MAXITER 100
#define DEFAULT_SEED 12345
#define DEFAULT_RF 100
#define DEFAULT_KERNEL 0
#define DEFAULT_FILEFORMAT 0

enum LossType {SQUARED = 0, LAD = 1, HINGE = 2, LOGISTIC = 3};
std::string Losses[] = {"Squared Loss",
                        "Least Absolute Deviations",
                        "Hinge Loss (SVMs)",
                        "Logistic Loss"};

enum RegularizerType {L2 = 0 , L1 = 1};
std::string Regularizers[] = {"L2", "L1"};

enum ProblemType {REGRESSION = 0, CLASSIFICATION = 1};
std::string Problems[] = {"Regression", "Classification"};

enum KernelType {LINEAR = 0, GAUSSIAN = 1, POLYNOMIAL = 2,
                 LAPLACIAN = 3, EXPSEMIGROUP = 4};
std::string Kernels[] = {"Linear", "Gaussian",
                         "Polynomial", "Laplacian", "ExpSemigroup"};

enum FileFormatType {LIBSVM = 0, HDF5 = 1};
std::string FileFormats[] = {"Libsvm", "HDF5"};

/**
 * A structure that is used to pass options to the ADMM solver. This structure
 * has default values embedded. No accessor functions are being written.
 */
struct hilbert_options_t {
    /** Solver type options */

    LossType lossfunction;
    RegularizerType regularizer;
    KernelType kernel;

    /** Kernel parameters */
    double kernelparam;
    double kernelparam2;
    double kernelparam3;

    double lambda;

    /** Optimization options */;
    int MAXITER;
    double tolerance;
    double rho;

    /** Randomization options */
    int seed;
    int randomfeatures;
    bool regularmap;

    /* parallelization options */
    int numfeaturepartitions;
    int numthreads;
    int nummpiprocesses;

    int fileformat;

    /**  IO */
    std::string trainfile;
    std::string modelfile;
    std::string testfile;
    std::string valfile;

    /** A parameter indicating if we need to continue or not */
    bool exit_on_return;

    /**
     * The constructor takes in all the command line parameters and parses them.
     */
    hilbert_options_t (int argc, char** argv, int nproc) :
        nummpiprocesses(nproc), exit_on_return(false) {

#ifndef SKYLARK_AVOID_BOOST_PO

        po::options_description desc
            ("Usage: hilbert_train [options] trainfile modelfile");
        desc.add_options()
            ("help,h", "produce a help message")
            ("lossfunction,l",
                po::value<int>((int*) &lossfunction)->default_value(SQUARED),
                "Loss function (0:SQUARED, 1:LAD, 2:HINGE, 3:LOGISTIC")
            ("regularizer,r",
                po::value<int>((int*) &regularizer)->default_value(L2),
                "Regularizer (0:L2, 1:L1)")
            ("kernel,k",
                po::value<int>((int*) &kernel)->default_value(LINEAR),
                "Kernel (0:LINEAR, 1:GAUSSIAN, 2:POLYNOMIAL, "
                "3:LAPLACIAN, 4:EXPSEMIGROUP)")
            ("kernelparam,g",
                po::value<double>(&kernelparam)->default_value(DEFAULT_KERPARAM),
                "Kernel Parameter")
            ("kernelparam2,x",
                po::value<double>(&kernelparam2)->default_value(-1),
                "If Applicable - Second Kernel Parameter (Polynomial Kernel: c)")
            ("kernelparam3,y",
                po::value<double>(&kernelparam3)->default_value(-1),
                "If Applicable - Third Kernel Parameter (Polynomial Kernel: gamma)")
            ("lambda,c",
                po::value<double>(&lambda)->default_value(DEFAULT_LAMBDA),
                "Regularization Parameter")
            ("tolerance,e",
                po::value<double>(&tolerance)->default_value(DEFAULT_TOL),
                "Tolerance")
            ("rho",
                po::value<double>(&rho)->default_value(DEFAULT_RHO),
                "ADMM rho parameter")
            ("seed,s",
                po::value<int>(&seed)->default_value(DEFAULT_SEED),
                "Seed for Random Number Generator")
            ("randomfeatures,f",
                po::value<int>(&randomfeatures)->default_value(DEFAULT_RF),
                "Number of Random Features (default: 100)")
            ("numfeaturepartitions,n",
                po::value<int>(&numfeaturepartitions)->
                default_value(DEFAULT_FEATURE_PARTITIONS),
                "Number of Feature Partitions (default: 1)")
            ("numthreads,t",
                po::value<int>(&numthreads)->default_value(DEFAULT_THREADS),
                "Number of Threads (default: 1)")
            ("regular",
                po::value<bool>(&regularmap)->default_value(false),
                "Default is to use 'fast' feature mapping, if available."
                "Use this flag to force regular mapping (default: false)")
            ("fileformat",
                po::value<int>(&fileformat)->default_value(DEFAULT_FILEFORMAT),
                "Fileformat (default: 0 (libsvm), 1 (hdf5)")
            ("MAXITER,i",
                po::value<int>(&MAXITER)->default_value(DEFAULT_MAXITER),
                "Maximum Number of Iterations (default: 100)")
            ("trainfile",
                po::value<std::string>(&trainfile)->required(),
                "Training data file")
            ("modelfile",
                po::value<std::string>(&modelfile)->required(),
                "Model output file")
            ("valfile",
                po::value<std::string>(&valfile)->default_value(""),
                "Validation file (optional)")
            ("testfile",
                po::value<std::string>(&testfile)->default_value(""),
                "Test file (optional)")
            ; /* end options */

        po::positional_options_description positionalOptions;
        positionalOptions.add("trainfile", 1);
        positionalOptions.add("modelfile", 1);

        /** create a variable map to hold all these things */
        po::variables_map vm;
        try {
            po::store(po::command_line_parser(argc, argv)
                .options(desc).positional(positionalOptions).run(), vm);

            /** Print help and return if needed */
            if (vm.count ("help")) {
                std::cout << desc;
                exit_on_return = true;
                return;
            }
            po::notify(vm); // throws on error, so do after help in case
            // there are any problems
        }
        catch(po::error& e) {
            std::cerr << e.what() << std::endl;
            std::cerr << desc << std::endl;
            exit_on_return = true;
            return;
        }

#else
        // The following is much less robust, but should work even without
        // Boost::program_options.

        lossfunction = SQUARED;
        regularizer = L2;
        kernel = LINEAR;
        kernelparam = DEFAULT_KERPARAM;
        kernelparam2 = -1;
        kernelparam3 = -1;
        lambda = DEFAULT_LAMBDA;
        tolerance = DEFAULT_TOL;
        rho = DEFAULT_RHO;
        seed = DEFAULT_SEED;
        randomfeatures = DEFAULT_RF;
        numfeaturepartitions = DEFAULT_FEATURE_PARTITIONS;
        numthreads = DEFAULT_THREADS;
        regularmap = false;
        fileformat = DEFAULT_FILEFORMAT;
        MAXITER = DEFAULT_MAXITER;
        valfile = "";
        testfile = "";

        for (int i = 1; i < argc; i += 2) {
            std::string flag = argv[i];
            std::string value = argv[i+1];

            if (flag == "--lossfunction" || flag == "-l")
                lossfunction =
                    static_cast<LossType>(boost::lexical_cast<int>(value));
            if (flag == "--regularizer" || flag == "-r")
                regularizer =
                    static_cast<RegularizerType>(boost::lexical_cast<int>(value));
            if (flag == "--kernel" || flag == "-k")
                kernel =
                    static_cast<KernelType>(boost::lexical_cast<int>(value));
            if (flag == "--kernelparam" || flag == "-g")
                kernelparam = boost::lexical_cast<double>(value);
            if (flag == "--kernelparam2" || flag == "-x")
                kernelparam2 = boost::lexical_cast<double>(value);
            if (flag == "--kernelparam3" || flag == "-y")
                kernelparam3 = boost::lexical_cast<double>(value);
            if (flag == "--lambda" || flag == "-c")
                lambda = boost::lexical_cast<double>(value);
            if (flag == "--tolerance" || flag == "-e")
                tolerance = boost::lexical_cast<double>(value);
            if (flag == "--rho")
                rho = boost::lexical_cast<double>(value);
            if (flag == "--seed" || flag == "-s")
                seed = boost::lexical_cast<int>(value);
            if (flag == "--randomfeatures" || flag == "-f")
                randomfeatures = boost::lexical_cast<int>(value);
            if (flag == "--numfeaturepartitions" || flag == "-n")
                numfeaturepartitions = boost::lexical_cast<int>(value);
            if (flag == "--numthreads" || flag == "-t")
                numthreads = boost::lexical_cast<int>(value);
            if (flag == "--regular")
                regularmap = value == "on";
            if (flag == "--fileformat")
                fileformat =
                    static_cast<FileFormatType>(boost::lexical_cast<int>(value));
            if (flag == "--MAXITER" || flag == "-i")
                MAXITER = boost::lexical_cast<int>(value);
            if (flag == "--trainfile")
                trainfile = value;
            if (flag == "--modelfile")
                modelfile = value;
            if (flag == "--valfile")
                valfile = value;
            if (flag == "--testfile")
                testfile = value;
        }
#endif
    }

    std::string print () const {
        std::stringstream optionstring;

        optionstring << "# HILBERT OPTIONS:" << std::endl;
        optionstring << "# Training File = " << trainfile << std::endl;
        optionstring << "# Model File = " << modelfile << std::endl;
        optionstring << "# Validation File = " << valfile << std::endl;
        optionstring << "# Test File = " << testfile << std::endl;
        optionstring << "# File Format = " << fileformat << std::endl;
        optionstring << "# Loss function = " << lossfunction
                     << " ("<< Losses[lossfunction]<< ")" << std::endl;
        optionstring << "# Regularizer = " << regularizer
                     << " ("<< Regularizers[regularizer]<< ")" << std::endl;
        optionstring << "# Kernel = " << kernel
                     << " ("<< Kernels[kernel]<< ")" << std::endl;
        optionstring << "# Kernel Parameter = " << kernelparam << std::endl;
        if (kernelparam2 != -1)
            optionstring << "# Second Kernel Parameter = "
                         << kernelparam2 << std::endl;
        if (kernelparam3 != -1)
            optionstring << " Third Kernel Parameter = "
                         << kernelparam3 << std::endl;
        optionstring << "# Regularization Parameter = " << lambda << std::endl;
        optionstring << "# Maximum Iterations = " << MAXITER << std::endl;
        optionstring << "# Tolerance = " << tolerance << std::endl;
        optionstring << "# rho = " << rho << std::endl;
        optionstring << "# Seed = " << seed << std::endl;
        optionstring << "# Random Features = " << randomfeatures << std::endl;
        optionstring << "# Number of feature partitions = "
                     << numfeaturepartitions << std::endl;
        optionstring << "# Threads = " << numthreads << std::endl;
        optionstring <<"# Number of MPI Processes = "
                     << nummpiprocesses << std::endl;

        return optionstring.str();
    }
};


#endif /* SKYLARK_HILBERT_OPTIONS_HPP */
