#include <El.hpp>

#include <boost/program_options.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

#include <iostream>

namespace bpo = boost::program_options;

enum file_types {
    LIBSVM,
    ARC_LIST
};

template<typename InputType, typename FactorType, typename UType = FactorType,
         typename YType = FactorType>
void execute(bool directory, const std::string &fname,
    const std::string &hdfs, int port, const std::vector<int> &profile, int k,
    const skylark::nla::approximate_svd_params_t &params,
    const std::string &prefix,
    skylark::base::context_t &context) {

    boost::mpi::communicator world;
    int rank = world.rank();

    InputType A;
    FactorType S, V;
    UType U;
    YType Y;

    boost::mpi::timer timer;

    if (profile.empty()) {
        // Load A and Y (Y is thrown away)
        if (rank == 0) {
            std::cout << "Reading the matrix... ";
            std::cout.flush();
            timer.restart();
        }

        if (!hdfs.empty()) {
#       if SKYLARK_HAVE_LIBHDFS

            hdfsFS fs;
            if (rank == 0)
                fs = hdfsConnect(hdfs.c_str(), port);

            if (directory)
                SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                    skylark::base::error_msg(
                                             "HDFS directory reading not yet supported."))
                else
                    skylark::utility::io::ReadLIBSVM(
                                                     fs, fname, A, Y, skylark::base::ROWS);

#       else

            SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                skylark::base::error_msg("Install libhdfs for HDFS support!"));

#       endif
        } else {
            if (directory)
                skylark::utility::io::ReadDirLIBSVM(
                                                    fname, A, Y, skylark::base::ROWS);
            else
                skylark::utility::io::ReadLIBSVM(fname, A, Y, skylark::base::ROWS);
        }

        Y.Empty();
    } else {

        if (rank == 0) {
            std::cout << "Generating random matrix... ";
            std::cout.flush();
            timer.restart();
        }

        skylark::base::UniformMatrix(A, profile[0], profile[1], context);
    }

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

template<typename InputType, typename FactorType, typename UType = FactorType,
         typename YType = FactorType>
void execute_sym(bool directory, const std::string &fname,
    const std::string& ftype,
    const std::string &hdfs, int port,
    const std::vector<int> &profile, bool lower, int k,
    const skylark::nla::approximate_svd_params_t &params,
    const std::string &prefix,
    skylark::base::context_t &context) {

    boost::mpi::communicator world;
    int rank = world.rank();

    InputType A;
    FactorType S, V;
    YType Y;

    boost::mpi::timer timer;

    if (profile.empty()) {
        // Load A and Y (Y is thrown away)
        if (rank == 0) {
            std::cout << "Reading the matrix... ";
            std::cout.flush();
            timer.restart();
        }

        if (!hdfs.empty()) {
#       if SKYLARK_HAVE_LIBHDFS

            hdfsFS fs;
            if (rank == 0)
                fs = hdfsConnect(hdfs.c_str(), port);

            if (directory)
                SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                    skylark::base::error_msg(
                                             "HDFS directory reading not yet supported."))
                else
                    skylark::utility::io::ReadLIBSVM(
                                                     fs, fname, A, Y, skylark::base::ROWS);

#       else

            SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
                skylark::base::error_msg("Install libhdfs for HDFS support!"));

#       endif
        } else {
            // FIXME: ugly, fix options
            if (ftype.compare("ARC_LIST") == 0) {
                skylark::utility::io::ReadArcList(fname, A, world, true);
            } else {
                if (directory)
                    skylark::utility::io::ReadDirLIBSVM(
                                                        fname, A, Y, skylark::base::ROWS);
                else
                    skylark::utility::io::ReadLIBSVM(
                                                     fname, A, Y, skylark::base::ROWS);
            }
        }

        Y.Empty();
        
    } else {

        SKYLARK_THROW_EXCEPTION(skylark::base::unsupported_base_operation() <<
            skylark::base::error_msg("Uniform symmetric matrix generating not supported yet."));
    }

    if (rank == 0)
        std::cout << "took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    /* Compute approximate SVD */
    if (rank == 0) {
        std::cout << "Computing approximate SVD...";
        std::cout.flush();
        timer.restart();
    }

    skylark::nla::ApproximateSymmetricSVD(lower ? El::LOWER : El::UPPER,
        A, V, S, k, context, params);

    if (rank == 0)
        std::cout <<"Took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    /* Write results */
    if (rank == 0) {
        std::cout << "Writing results...";
        std::cout.flush();
        timer.restart();
    }

    El::Write(S, prefix + ".S", El::ASCII);
    El::Write(V, prefix + ".V", El::ASCII);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    world.barrier();
}


int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();

    int seed, k, powerits, port;
    std::string fname, ftype, prefix, hdfs;
    bool as_symmetric, as_sparse, skipqr, use_single, lower, directory;
    int oversampling_ratio, oversampling_additive;
    std::vector<int> profile;
    
    // Parse options
    bpo::options_description desc("Options");
    desc.add_options()
        ("help,h", "produce a help message")
        ("inputfile",
            bpo::value<std::string>(&fname),
            "Input file to run approximate SVD on (default libsvm format).")
        ("filetype",
            bpo::value<std::string>(&ftype),
            "Input file type (LIBSVM or ARC_LIST).")
        ("directory,d", "Whether inputfile is a directory of files whose"
            " concatination is the input.")
        ("seed,s",
            bpo::value<int>(&seed)->default_value(38734),
            "Seed for random number generation. OPTIONAL.")
        ("hdfs",
            bpo::value<std::string>(&hdfs)->default_value(""),
            "If not empty, will assume file is in an HDFS. "
            "Parameter is filesystem name.")
        ("port",
            bpo::value<int>(&port)->default_value(0),
            "For HDFS: port to use.")
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
        ("symmetric", "Whether to treat the matrix as symmetric. "
         "Only upper part will be accessed, unless --lower is used.")
        ("lower", "For symmetric matrix, access only lower part (upper is default).")
        ("sparse", "Whether to load the matrix as a sparse one.")
        ("single", "Whether to use single precision instead of double.")
        ("profile",
            bpo::value<std::vector<int> >()->multitoken(),
            "Generate random matrix and run on it (for profiling)."
            "Requires specification of height and width. OPTIONAL.")
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
            if (rank == 0) {
                std::cout << "Usage: " << argv[0]
                          << " [options] input-file-name" << std::endl;
                std::cout << desc;
            }
            world.barrier();
            return 0;
        }

        if (vm["profile"].empty() && !vm.count("inputfile")) {
            if (rank == 0)
                std::cout << "Input file is required." << std::endl;
            world.barrier();
            return -1;
        }

        if (!vm["profile"].empty()) {
            if (vm["profile"].as<std::vector<int> >().size() < 2) {
                if (rank == 0)
                    std::cout << "Please specify height and width for --profile."
                              << std::endl;
                world.barrier();
                return -1;
            } else
                profile = vm["profile"].as<std::vector<int> >();
        }
        
        bpo::notify(vm);

        as_symmetric = vm.count("symmetric");
        as_sparse = vm.count("sparse");
        skipqr = vm.count("skipqr");
        use_single = vm.count("single");
        directory = vm.count("directory");
        lower = vm.count("lower");

    } catch(bpo::error& e) {
        if (rank == 0) {
            std::cerr << e.what() << std::endl;
            std::cerr << desc << std::endl;
        }
        world.barrier();
        return -1;
    }

    skylark::base::context_t context(seed);

    skylark::nla::approximate_svd_params_t params;
    params.skip_qr = skipqr;
    params.num_iterations = powerits;
    params.oversampling_ratio = oversampling_ratio;
    params.oversampling_additive = oversampling_additive;

    SKYLARK_BEGIN_TRY()

        if (size == 1) {
            if (!as_symmetric) {

                if (use_single) {
                    if (as_sparse)
                        execute<skylark::base::sparse_matrix_t<float>,
                                El::Matrix<float> >(
                                    directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);
                    else
                        execute<El::Matrix<float>,
                                El::Matrix<float> >(directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);

                } else {
                    if (as_sparse)
                        execute<skylark::base::sparse_matrix_t<double>,
                                El::Matrix<double> >(
                                     directory, fname, hdfs,
                                     port, profile, k, params, prefix, context);
                    else
                        execute<El::Matrix<double>,
                                El::Matrix<double> >(directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);
                }

            } else {

                if (use_single) {
                    if (as_sparse)
                        execute_sym<skylark::base::sparse_matrix_t<float>,
                                    El::Matrix<float> >(
                                       directory, fname, ftype,
                                       hdfs, port, profile, lower, k, params, prefix,
                                        context);
                    else
                        execute_sym<El::Matrix<float>,
                                    El::Matrix<float> >(directory, fname, ftype,
                                        hdfs, port, profile, lower, k, params, prefix,
                                        context);

                } else {
                    if (as_sparse)
                        execute_sym<skylark::base::sparse_matrix_t<double>,
                                    El::Matrix<double>,
                                    El::Matrix<double> >(
                                        directory, fname, ftype, hdfs, port,
                                        profile, lower, k, params, prefix, context);
                    else
                        execute_sym<El::Matrix<double>,
                                    El::Matrix<double> >(directory, fname,
                                        ftype, hdfs, port, profile, lower, k, params,
                                        prefix, context);
                }
            }

        } else {
            if (!as_symmetric) {

                if (use_single) {
                    if (as_sparse)
                        execute<skylark::base::sparse_vc_star_matrix_t<float>,
                                El::DistMatrix<float, El::STAR, El::STAR>,
                                El::DistMatrix<float, El::VC, El::STAR>,
                                El::DistMatrix<float, El::VC, El::STAR> >(
                                    directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);
                    else
                        execute<El::DistMatrix<float>,
                                El::DistMatrix<float> >(directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);

                } else {
                    if (as_sparse)
                        execute<skylark::base::sparse_vc_star_matrix_t<double>,
                                El::DistMatrix<double, El::STAR, El::STAR>,
                                El::DistMatrix<double, El::VC, El::STAR>,
                                El::DistMatrix<double, El::VC, El::STAR> >(
                                    directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);
                    else
                        execute<El::DistMatrix<double>,
                                El::DistMatrix<double> >(directory, fname, hdfs,
                                    port, profile, k, params, prefix, context);
                }

            } else {

                if (use_single) {
                    if (as_sparse)
                        execute_sym<skylark::base::sparse_vc_star_matrix_t<float>,
                                    El::DistMatrix<float, El::VC, El::STAR>,
                                    El::DistMatrix<float, El::VC, El::STAR> >(
                                        directory, fname, ftype,
                                        hdfs, port, profile, lower, k, params, prefix,
                                        context);
                    else
                        execute_sym<El::DistMatrix<float>,
                                    El::DistMatrix<float> >(directory, fname, ftype,
                                        hdfs, port, profile, lower, k, params, prefix,
                                        context);

                } else {
                    if (as_sparse)
                        execute_sym<skylark::base::sparse_vc_star_matrix_t<double>,
                                    El::DistMatrix<double, El::VC, El::STAR>,
                                    El::DistMatrix<double, El::VC, El::STAR> >(
                                        directory, fname, ftype, hdfs, port,
                                        profile, lower,
                                        k, params, prefix, context);
                    else
                        execute_sym<El::DistMatrix<double>,
                                    El::DistMatrix<double> >(directory, fname,
                                        ftype, hdfs, port, profile, lower, k, params,
                                        prefix, context);
                }
            }
        }


    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT((rank == 0))

    El::Finalize();
    return 0;
}
