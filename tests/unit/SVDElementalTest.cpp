#include <iostream>
#include "boost/program_options.hpp"

#include "config.h"
#include "test_utils.hpp"

namespace po = boost::program_options;

int test_main(int argc, char* argv[]) {
    El::Initialize(argc, argv);

    int height;
    int width;

    // Declare the supported options.
    po::options_description desc("Options");
    desc.add_options()
        ("help", "help message")
        ("height", po::value<int>(&height)->default_value(10),
            "height of input matrix")
        ("width", po::value<int>(&width)->default_value(6),
            "width of input matrix")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("height")) {
        std::cout << "Height of input matrix was set to "
             << vm["height"].as<int>() << ".\n";
    }

    if (vm.count("width")) {
        std::cout << "Width of input matrix was set to "
             << vm["width"].as<int>() << ".\n";
    }

    // Declare distributed matrices
    El::DistMatrix<double> A;
    El::DistMatrix<double, El::VC,   El::STAR>  A_VC_STAR;
    El::DistMatrix<double, El::STAR, El::VC>    A_STAR_VC;
    El::DistMatrix<double, El::VR,   El::STAR>  A_VR_STAR;
    El::DistMatrix<double, El::STAR, El::VR>    A_STAR_VR;

    El::Uniform(A, height, width);
    A_VC_STAR = A;
    A_VR_STAR = A;
    Transpose(A, A_STAR_VC);
    Transpose(A, A_STAR_VR);

    // Call tester for each case
    test::util::check(A);
    test::util::check(A_VC_STAR);
    test::util::check(A_STAR_VC);
    test::util::check(A_VR_STAR);
    test::util::check(A_STAR_VR);

    El::Finalize();
    return 0;
}
