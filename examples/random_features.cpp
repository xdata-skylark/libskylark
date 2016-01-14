#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

#ifndef SKYLARK_AVOID_BOOST_PO

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

int parse_program_options(int argc, char* argv[], 
    int &seed, int &s, int &mind, std::string &infile, std::string &outfile,
    double &sigma) {


    bpo::options_description desc("Options");
    desc.add_options()
        ("help,h", "produce a help message")
        ("input",
            bpo::value<std::string>(&infile),
            "Input file.")
        ("output",
            bpo::value<std::string>(&outfile),
            "Output file.")
        ("seed,s",
            bpo::value<int>(&seed),
            "Seed for random number generation. THIS DETERMINES THE TRANSFORM!.")
        ("sigma,x",
            bpo::value<double>(&sigma),
            "Kernel bandwidth.")
        ("mind,d",
            bpo::value<int>(&mind)->default_value(0),
            "Kernel bandwidth.")
        ("numfeatures,f",
            bpo::value<int>(&s),
            "Number of random features.");

    bpo::positional_options_description positional;
    positional.add("input", 1);
    positional.add("output", 2);

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0]
                      << " [options] input-file-name output-file-name"
                      << std::endl;
            std::cout << desc;
            return 0;
        }

        if (!vm.count("input") || !vm.count("output")) {
            std::cout << "Input and output files are required." << std::endl;
            return -1;
        }

        if (!vm.count("seed")) {
            std::cout << "Seed is required." << std::endl;
            return -1;
        }

        if (!vm.count("sigma")) {
            std::cout << "Sigma is required." << std::endl;
            return -1;
        }

        if (!vm.count("numfeatures")) {
            std::cout << "Number of features is required." << std::endl;
            return -1;
        }

        bpo::notify(vm);
    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    return 1000;
}


#else

// TODO

#endif

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    int seed, s, mind;
    std::string infile, outfile;
    double sigma;

    int flag = parse_program_options(argc, argv, seed, s, mind,
        infile, outfile, sigma);
    if (flag != 1000)
        return flag;

    skylark::base::context_t context(seed);

    boost::mpi::communicator world;
    int rank = world.rank();

    typedef El::DistMatrix<double, El::STAR, El::VR> data_matrix_t;
    typedef El::DistMatrix<El::Int, El::STAR, El::VR> label_matrix_t;

    data_matrix_t X;
    label_matrix_t L;

    boost::mpi::timer timer;

    // Load A and Y
    if (rank == 0) {
        std::cout << "Reading the matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::utility::io::ReadLIBSVM(infile, X, L,
        skylark::base::COLUMNS, mind);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    // Compute kernel matrix
    if (rank == 0) {
        std::cout << "Computing the transformed matrix... ";
        std::cout.flush();
        timer.restart();
    }

    data_matrix_t Z(s, X.Width());
    skylark::ml::gaussian_t k(X.Height(), sigma);
    auto S = k.create_rft<data_matrix_t, data_matrix_t>(s,
        skylark::ml::regular_feature_transform_tag(), context);
    S->apply(X, Z, skylark::sketch::columnwise_tag());

    if (rank == 0)
        std::cout << "took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    // Write output
    if (rank == 0) {
        std::cout << "Writing the matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::utility::io::WriteLIBSVM(outfile, Z, L, skylark::base::COLUMNS);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    El::Finalize();
    return 0;
}
