#include <iostream>
#include "boost/program_options.hpp"

#include "config.h"
#include "../../base/svd.hpp"
#include "test_utils.hpp"


namespace po = boost::program_options;

int test_main(int argc, char* argv[]) {
    elem::Initialize(argc, argv);

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
    elem::DistMatrix<double> A;
    elem::DistMatrix<double, elem::VC,   elem::STAR>  A_VC_STAR;
    elem::DistMatrix<double, elem::STAR, elem::VC>    A_STAR_VC;
    elem::DistMatrix<double, elem::VR,   elem::STAR>  A_VR_STAR;
    elem::DistMatrix<double, elem::STAR, elem::VR>    A_STAR_VR;

    elem::Uniform(A, height, width);
    A_VC_STAR = A;
    A_VR_STAR = A;
    Transpose(A, A_STAR_VC);
    Transpose(A, A_STAR_VR);

    // Call tester for each case
    check(A);
    check(A_VC_STAR);
    check(A_STAR_VC);
    check(A_VR_STAR);
    check(A_STAR_VR);

    elem::Finalize();
    return 0;
}
