#include <iostream>
#include <boost/program_options.hpp>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

namespace bpo = boost::program_options;

template<typename InputType, typename RhsType, typename SolType>
void execute(const std::string &fname, 
    const std::string &outname, bool high,
    skylark::base::context_t &context) {

    boost::mpi::communicator world;
    int rank = world.rank();

    InputType A;
    RhsType b;

    boost::mpi::timer timer;

    // Load A and Y (Y is thrown away)
    if (rank == 0) {
        std::cout << "Reading the matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::utility::io::ReadLIBSVM(fname, A, b, skylark::base::ROWS);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    /* Compute approximate SVD */
    if (rank == 0) {
        std::cout << "Solving the least squares...";
        std::cout.flush();
        timer.restart();
    }

    // TODO shouldn't sizes be set inside?
    El::Int m = skylark::base::Height(A);
    El::Int n = skylark::base::Width(A);
    El::Int k = skylark::base::Width(b);

    SolType x(n, k);
    if (high)
        skylark::nla::FastLeastSquares(El::NORMAL, A, b, x, context);
    else
        skylark::nla::ApproximateLeastSquares(El::NORMAL, A, b, x, context);

    if (rank == 0)
        std::cout <<"Took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    /* Write results */
    if (rank == 0) {
        std::cout << "Writing results...";
        std::cout.flush();
        timer.restart();
    }

    El::Write(x, outname, El::ASCII);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";
}

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    int seed;
    std::string fname, outfile;
    bool as_sparse, high, use_single;

    // Parse options
    bpo::options_description desc("Options");
    desc.add_options()
        ("help,h", "produce a help message")
        ("inputfile",
            bpo::value<std::string>(&fname),
            "Input file to run approximate SVD on (libsvm format).")
        ("seed,s",
            bpo::value<int>(&seed)->default_value(38734),
            "Seed for random number generation. OPTIONAL.")

        //("sparse", "Whether to load the matrix as a sparse one.")
        ("highprecision,p", "Solve to high precision.")
        ("single,f", "Whether to use single precision instead of double.")
        ("outputfile",
            bpo::value<std::string>(&outfile)->default_value("out"),
            "Prefix for output files (prefix.txt). OPTIONAL.");

    bpo::positional_options_description positional;
    positional.add("inputfile", 1);
    positional.add("outputfile", 2);

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options] input-file-name"
                      << std::endl;
            std::cout << desc;
            return 0;
        }

        if (!vm.count("inputfile")) {
            std::cout << "Input file is required." << std::endl;
            return -1;
        }

        bpo::notify(vm);

        //as_sparse = vm.count("sparse");
        use_single = vm.count("single");
        high = vm.count("highprecision");

    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    skylark::base::context_t context(seed);

;

    if (use_single) {
        // if (as_sparse)
        //     execute<skylark::base::sparse_matrix_t<float>,
        //             El::Matrix<float>,
        //             El::Matrix<float> >(fname, outfile, high, context);
        // else
            execute<El::DistMatrix<float>,
                    El::DistMatrix<float>,
                    El::DistMatrix<float> >(fname, outfile, high, context);

    } else {
        // if (as_sparse)
        //     execute<skylark::base::sparse_matrix_t<double>,
        //             El::Matrix<double>,
        //             El::Matrix<double> >(fname, outfile, high, context);
        // else
            execute<El::DistMatrix<double>,
                    El::DistMatrix<double>,
                    El::DistMatrix<double> >(fname, outfile, high,  context);
    }

    El::Finalize();
    return 0;
}
