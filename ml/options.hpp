/*
 * options.hpp
 *
 *  Created on: Jan 29, 2014
 *      Author: vikas
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <boost/program_options.hpp>

namespace po = boost::program_options;

#define DEFAULT_LAMBDA 0.0
#define DEFAULT_RHO 1.0
#define DEFAULT_THREADS 1
#define DEFAULT_FEATURE_PARTITIONS 1
#define DEFAULT_KERNELPARAM 1.0
#define DEFAULT_TOL 0.001
#define DEFAULT_MAXITER 100
#define DEFAULT_SEED 12345
#define DEFAULT_RF 100
#define DEFAULT_KERNEL 0
#define DEFAULT_FILEFORMAT 0

enum LossType {SQUARED = 0, LAD = 1, HINGE = 2, LOGISTIC = 3};
std::string Losses[] = {"Squared Loss", "Least Absolute Deviations", "Hinge Loss (SVMs)", "Logistic Loss"};

enum RegularizerType {L2 = 0 , L1 = 1};
std::string Regularizers[] = {"L2", "L1"};

enum ProblemType {REGRESSION = 0, CLASSIFICATION = 1};
std::string Problems[] = {"Regression", "Classification"};

enum KernelType {LINEAR = 0, GAUSSIAN = 1};
std::string Kernels[] = {"Linear", "Gaussian"};

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
 // ProblemType problem;

 /** Kernel parameters */
 double kernelparam;
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

 /* acceleration options */
 // bool fastfood;
 //bool rounding;

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
  hilbert_options_t (int argc, char** argv, int nproc) : exit_on_return(false) {
    /** Set up the options that we want */
      this->nummpiprocesses = nproc;
    po::options_description desc ("Usage: hilbert_train [options] trainfile modelfile");
    desc.add_options()
  ("help,h", "produce a help message")
  ("lossfunction,l", po::value<int>((int*) &lossfunction)->default_value(SQUARED), "Loss function (0:SQUARED, 1:LAD, 2:HINGE, 3:LOGISTIC")
  ("regularizer,r",    po::value<int>((int*) &regularizer)->default_value(L2), "Regularizer (0:L2, 1:L1)")
  ("kernel,k", po::value<int>((int*) &kernel)->default_value(LINEAR), "Kernel (0:LINEAR, 1:GAUSSIAN)")
  ("kernelparam,g", po::value<double>(&kernelparam)->default_value(DEFAULT_KERNELPARAM), "Kernel Parameter")
  ("lambda,c", po::value<double>(&lambda)->default_value(DEFAULT_LAMBDA), "Regularization Parameter")
  ("tolerance,e", po::value<double>(&tolerance)->default_value(DEFAULT_TOL), "Tolerance")
  ("rho", po::value<double>(&rho)->default_value(DEFAULT_RHO), "ADMM rho parameter")
  ("seed,s", po::value<int>(&seed)->default_value(DEFAULT_SEED), "Seed for Random Number Generator")
  ("randomfeatures,f", po::value<int>(&randomfeatures)->default_value(DEFAULT_RF), "Number of Random Features (default: 100)")
  ("numfeaturepartitions,n", po::value<int>(&numfeaturepartitions)->default_value(DEFAULT_FEATURE_PARTITIONS), "Number of Feature Partitions (default: 1)")
  ("numthreads,t", po::value<int>(&numthreads)->default_value(DEFAULT_THREADS), "Number of Threads (default: 1)")
  ("regular", po::value<bool>(&regularmap)->default_value(false), "Default is to use 'fast' feature mapping, if available. Use this flag to force regular mapping (default: false)")
  ("fileformat", po::value<int>(&fileformat)->default_value(DEFAULT_FILEFORMAT), "Fileformat (default: 0 (libsvm), 1 (hdf5)")
  ("MAXITER,i", po::value<int>(&MAXITER)->default_value(DEFAULT_MAXITER), "Maximum Number of Iterations (default: 100)")
  ("trainfile", po::value<std::string>(&trainfile)->required(), "Training data file")
  ("modelfile", po::value<std::string>(&modelfile)->required(), "Model output file")
  ("valfile", po::value<std::string>(&valfile)->default_value(""), "Validation file (optional)")
  ("testfile", po::value<std::string>(&testfile)->default_value(""), "Test file (optional)")
      ; /* end options */

    po::positional_options_description positionalOptions;
    positionalOptions.add("trainfile", 1);
    positionalOptions.add("modelfile", 1);

    /** create a variable map to hold all these things */
    po::variables_map vm;
    try {
    		po::store(po::command_line_parser(argc, argv).options(desc).positional(positionalOptions).run(), vm);

    /** Print help and return if needed */
    	if (vm.count ("help")) {
    		std::cout << desc;
    		exit_on_return = true;
    		return;
    	}
    	 po::notify(vm); // throws on error, so do after help in case
    	                          // there are any problems
    	}
    	catch(po::error& e)
    	        {
    		      std::cerr << e.what() << std::endl;
    	          std::cerr << desc << std::endl;
    	          exit_on_return = true;
    	          return;
    	        }
  }

  std::string print () const {
	 std::stringstream optionstring;
	 optionstring << "# HILBERT OPTIONS:" << "\n";
	 optionstring << "# Training File = " << trainfile << "\n";
	 optionstring << "# Model File = " << modelfile << "\n";
	 optionstring << "# Validation File = " << valfile << "\n";
	 optionstring << "# Test File = " << testfile << "\n";
	 optionstring << "# File Format = " << fileformat << "\n";
	 optionstring << "# Loss function = " << lossfunction << " ("<< Losses[lossfunction]<< ")" << "\n";
	 optionstring << "# Regularizer = " << regularizer << " ("<< Regularizers[regularizer]<< ")" << "\n";
	 optionstring << "# Kernel = " << kernel << " ("<< Kernels[kernel]<< ")" << "\n";
	 optionstring << "# Kernel Parameter = " << kernelparam << "\n";
	 optionstring << "# Regularization Parameter = " << lambda << "\n";
	 optionstring << "# Maximum Iterations = " << MAXITER << "\n";
	 optionstring << "# Tolerance = " << tolerance << "\n";
	 optionstring << "# rho = " << rho << "\n";
	 optionstring << "# Seed = " << seed << "\n";
	 optionstring << "# Random Features = " << randomfeatures << "\n";
	 optionstring << "# Number of feature partitions = " << numfeaturepartitions << "\n";
	 optionstring << "# Threads = " << numthreads << "\n";
	 optionstring <<"# Number of MPI Processes = " << nummpiprocesses << "\n";
	 return optionstring.str();
  }
};


#endif /* OPTIONS_HPP_ */
