/*
 * options.hpp
 *
 *  Created on: Jan 29, 2014
 *      Author: vikas
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <boost/program_options.hpp>
#include "hilbert.hpp"

namespace po = boost::program_options;

namespace po = boost::program_options;

/**
 * A structure that is used to pass options to the ADMM solver. This structure
 * has default values embedded. No accessor functions are being written.
 */
struct hilbert_options_t {
  /** Solver type options */

 LossType lossfunction;
 RegularizerType regularizer;
 KernelType kernel;
 ProblemType problem;

 /** Kernel parameters */
 double kernelparam;

 /** Optimization options */;
 int MAXITER;
 float tolerance;

 /** Randomization options */
 int seed;
 int randomfeatures;

 /* parallelization options */
 int numfeaturepartitions;
 int numthreads;

 /* acceleration options */
 bool fastfood;
 bool rounding;

 /**  IO */
 std::string trainfile;
 std::string modelfile;

  /** A parameter indicating if we need to continue or not */
 bool exit_on_return;

/**
   * The constructor takes in all the command line parameters and parses them.
   */
  hilbert_options_t (int argc, char** argv) : exit_on_return(false) {
    /** Set up the options that we want */
    po::options_description desc ("Hilbert Options");
    desc.add_options()
  ("help", "produce a help message")
  ("n-max", po::value<int>(&n_max)->default_value(10),
                                       "Maximum number of nodes in the tree")
  ("n-min", po::value<int>(&n_min)->default_value(3),
                                       "Minimum number of nodes in the tree")
  ("seed", po::value<int>(&seed)->default_value(-1), "Random number seed")
  ("mle-seed", po::value<int>(&mle_seed)->default_value(-1),
               "Random number seed for uniformly picked vertex for MLE")
  ("verbosity", po::value<int>(&verbosity)->default_value(1),
   "Verbosity level for error messages")
  ("k", po::value<int>(&k)->default_value(4), "Number of Bernoulli trials")
  ("lambda", po::value<int>(&lambda)->default_value(10), "Mean for Poisson")
  ("num-trials", po::value<int>(&num_trials)->default_value(1),
                              "Number of trials to run for this experiment")
  ("num-threads", po::value<int>(&num_threads)->default_value(4),
                              "Number of threads to use for this experiment")
  ("batch-size", po::value<int>(&batch_size)->default_value(10),
                              "Batch size for testing confidence")
  ("use-random-weights", po::value<bool>(&use_random_weights)
                         ->default_value(false),
   "Should we have random edge weights (between 0 and 1): false)")
  ("print-tree", po::value<bool>(&print_tree)
                       ->default_value(false),
   "Should we print the tree as a GraphViz graph: false)")
  ("print-path", po::value<bool>(&print_path)
                       ->default_value(false),
   "Should we print the Dyck path of the tree: false)")
  ("measure-mle", po::value<bool>(&measure_mle)
                       ->default_value(false),
   "Should we measure the MLE of a randomly picked vertex in the tree: false)")
  ("test-confidence", po::value<bool>(&test_confidence)
                       ->default_value(false),
   "Should we measure the CI a randomly picked vertex in the tree: false)")
  ("dump-numbers", po::value<bool>(&dump_numbers)
                       ->default_value(false),
   "Should we dump mean heights and sums of squares: false)")
  ("p", po::value<double>(&p)->default_value(0.01),
   "Probability of success for the Bernoulli trials (default: 0.01)")
  ("gen-method", po::value<std::string>(&gen_method)->default_value("Poisson"),
   "Distribution to use for generating the tree (default:Poisson)")
  ("dot-out", po::value<std::string>(&dot_out)->default_value("stdout"),
   "File name to print the tree GraphViz out")
  ("dyck-out", po::value<std::string>(&dyck_out)->default_value("stdout"),
   "File name to print the tree Dyck path out")
  ("response-file", po::value<std::string>(),
                          "can be specified with '@name', too")
      ; /* end options */

    /** create a variable map to hold all these things */
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc)
              .extra_parser(at_option_parser).run(), vm);

    /** Print help and return if needed */
    if (vm.count ("help")) {
      std::cout << desc;
      exit_on_return = true;
      return;
    }

    try {
      if (vm.count("response-file")) {
        /** Load the file and tokenize it */
        std::ifstream ifs (vm["response-file"].as<std::string>().c_str());
        if (false==ifs) {
          std::cout << "Could not open response file" << std::endl;
          exit_on_return = true;
          return;
        }

        /** Read the whole file into a string */
        std::stringstream ss;
        ss << ifs.rdbuf ();
        boost::char_separator<char> sep(" \n\r");
        std::string sstr(ss.str());
        boost::tokenizer<boost::char_separator<char> > tok(sstr, sep);
        std::vector<std::string> args;
        std::copy (tok.begin(), tok.end(), std::back_inserter(args));

        /** Parse the file and store the options */
        po::store(po::command_line_parser(args).options(desc).run(), vm);
      }

      po::notify(vm);
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
      exit_on_return = true;
    }

    /** Now, check for error conditions */
    if (0 == strcmp("Poisson", gen_method.c_str())) {
    } else if (0 == strcmp("Binomial", gen_method.c_str())) {
    } else if (0 == strcmp("Geometric", gen_method.c_str())) {
    } else if (0 == strcmp("Binary-0-2", gen_method.c_str())) {
    } else if (0 == strcmp("Binary-0-1-2", gen_method.c_str())) {
    } else {
      std::cout << "Graph generation not supported" << std::endl;
    }
  }

  void pretty_print () const {
    std::cout << "n_max = " << n_max << std::endl;
    std::cout << "n_min = " << n_min << std::endl;
    std::cout << "seed = " << seed << std::endl;
    std::cout << "MLE-seed = " << mle_seed << std::endl;
    std::cout << "verbosity = " << verbosity << std::endl;
    std::cout << "k = " << k << std::endl;
    std::cout << "lambda = " << lambda << std::endl;
    std::cout << "p = " << p << std::endl;
    std::cout << "num-trials = " << num_trials << std::endl;
    std::cout << "num-threads = " << num_threads << std::endl;
    std::cout << "batch-size = " << batch_size << std::endl;
    std::cout << "gen-method = " << gen_method << std::endl;
    std::cout << "Use random weights = " << use_random_weights << std::endl;
    std::cout << "Print tree = " << print_tree << std::endl;
    std::cout << "Print Dyck path = " << print_path << std::endl;
    std::cout << "Estimate MLE measure = " << measure_mle << std::endl;
    std::cout << "Estimate confidence = " << test_confidence << std::endl;
    std::cout << "Tree file = " << dot_out << std::endl;
    std::cout << "Path file = " << dyck_out << std::endl;
  }
};




#endif /* OPTIONS_HPP_ */
