#include <iostream>
#include <boost/program_options.hpp>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

namespace bpo = boost::program_options;

template<typename InputType, typename FactorType>
void execute(const std::string &fname, int k,
    const skylark::nla::approximate_svd_params_t &params,
    const std::string &prefix,
    skylark::base::context_t &context) {

    boost::mpi::communicator world;
    int rank = world.rank();

    InputType A;
    FactorType U, S, V, Y;

    boost::mpi::timer timer;

    // Load A and Y (Y is thrown away)
    if (rank == 0) {
        std::cout << "Reading the matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::utility::io::ReadLIBSVM(fname, A, Y, skylark::base::ROWS);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    /* Compute approximate SVD */
    if (rank == 0) {
        std::cout << "Computing approximate SVD...";
        std::cout.flush();
        timer.restart();
    }

    skylark::nla::ApproximateSVD(A, U, S, V, k, context, params);

    if (rank == 0)
        std::cout <<"Took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    /* Write results */
    if (rank == 0) {
        std::cout << "Writing results...";
        std::cout.flush();
        timer.restart();
    }

    El::Write(U, prefix + ".U", El::ASCII);
    El::Write(S, prefix + ".S", El::ASCII);
    El::Write(V, prefix + ".V", El::ASCII);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";
}

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();

    int seed, k, powerits;
    std::string fname, prefix;
    bool as_sparse, skipqr, use_single;
    int oversampling_ratio, oversampling_additive;

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
        ("rank,k",
            bpo::value<int>(&k)->default_value(6),
            "Target rank. OPTIONAL.")
        ("powerits,i",
            bpo::value<int>(&powerits)->default_value(2),
            "Number of power iterations. OPTIONAL.")
        ("skipqr", "Whether to skip QR in each iteration. Higher than one power"
            " iterations is not recommended in this mode.")
        ("ratio,r",
            bpo::value<int>(&oversampling_ratio)->default_value(2),
            "Ratio of oversampling of rank. OPTIONAL.")
        ("additive,a",
            bpo::value<int>(&oversampling_additive)->default_value(0),
            "Additive factor for oversampling of rank. OPTIONAL.")
        ("sparse", "Whether to load the matrix as a sparse one.")
        ("single", "Whether to use single precision instead of double.")
        ("prefix",
            bpo::value<std::string>(&prefix)->default_value("out"),
            "Prefix for output files (prefix.U.txt, prefix.S.txt"
            " and prefix.V.txt. OPTIONAL.");

    bpo::positional_options_description positional;
    positional.add("inputfile", 1);

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

        as_sparse = vm.count("sparse");
        skipqr = vm.count("skipqr");
        use_single = vm.count("single");

    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    skylark::base::context_t context(seed);

    skylark::nla::approximate_svd_params_t params;
    params.skip_qr = skipqr;
    params.num_iterations = powerits;
    params.oversampling_ratio = oversampling_ratio;
    params.oversampling_additive = oversampling_additive;

    SKYLARK_BEGIN_TRY()

    if (use_single) {
        if (as_sparse)
            execute<skylark::base::sparse_matrix_t<float>,
                    El::Matrix<float> >(fname, k, params, prefix, context);
        else
            execute<El::DistMatrix<float>,
                    El::DistMatrix<float> >(fname, k, params, prefix, context);

    } else {
        if (as_sparse)
            execute<skylark::base::sparse_matrix_t<double>,
                    El::Matrix<double> >(fname, k, params, prefix, context);
        else
            execute<El::DistMatrix<double>,
                    El::DistMatrix<double> >(fname, k, params, prefix, context);
    }

    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT((rank == 0))

    El::Finalize();
    return 0;
}
