/**
 *  This test ensures that reading edge list files (matrix market) works as
 *  intended.
 */

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#include <skylark.hpp>

#include <functional>
#include <sstream>
#include <string>
#include <vector>

int test_main(int argc, char *argv[]) {
    //////////////////////////////////////////////////////////////////////////
    //[> Setup test <]

    namespace mpi = boost::mpi;
    mpi::environment env(argc, argv);
    mpi::communicator world;

    int num_partitions = world.size();

    //////////////////////////////////////////////////////////////////////////
    //[> Test Chunker <]
    //
    // We read the file in chunks and make sure all the data is read correctly
    // by exactly on rank.

    std::stringstream data;

    try {
        skylark::utility::io::detail::parallelChunkedRead(argv[1], world,
            num_partitions, data);
    } catch (skylark::base::skylark_exception ex) {
        SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
        SKYLARK_PRINT_EXCEPTION_TRACE(ex);
        errno = *(boost::get_error_info<skylark::base::error_code>(ex));
        std::cout << "Caught exception, exiting with error " << errno << ": ";
        std::cout << skylark_strerror(errno) << std::endl;
        BOOST_FAIL("Exception when reading chunks in parallel.");
    }

    // read the ref data
    std::stringstream ref_data;
    if (world.rank() == 0) {
        std::ifstream file(argv[1]);
        if (file) {
            ref_data << file.rdbuf();
            file.close();
        }
    }

    std::vector<std::string> vec_data;
    boost::mpi::gather(world, data.str(), vec_data, 0);

    if (world.rank() == 0) {
        // compare line by line for nicer error reporting
        ref_data.seekp(0);
        ref_data.seekg(0);
        size_t line_nr = 0;
        for (size_t i = 0; i < vec_data.size(); i++) {
            std::stringstream res_data;
            res_data << vec_data[i];

            std::string line;
            // XXX: we need to strip the first partial line for all ranks after
            //      0. Note that this is not necessarely true since a rank can
            //      get the start of a line.
            if (i > 0)
                std::getline(res_data, line);

            while (std::getline(res_data, line)) {
                std::string ref_line;
                std::getline(ref_data, ref_line);
                line_nr++;
                if (ref_data.eof() || !ref_data.good()) {
                    std::cout << "Line " << line_nr << std::endl;
                    BOOST_FAIL("Error: reference data stream went bad");
                }

                if (ref_line.compare(line) != 0) {
                    std::cout << "Line " << line_nr << ": "
                              << ref_line << " != " << line << std::endl;
                    BOOST_FAIL("Error in read");
                }
            }
        }
    }


    //////////////////////////////////////////////////////////////////////////
    //[> Test Reader <]
    //
    // Make sure the parsing and redistributing produces a valid matrix.

    El::Initialize(argc, argv);
    skylark::base::sparse_vc_star_matrix_t<double> A;

    try {
        std::string fname(argv[1]);
        skylark::utility::io::ReadArcList(fname, A, world, true);
    } catch (skylark::base::skylark_exception ex) {
        SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
        SKYLARK_PRINT_EXCEPTION_TRACE(ex);
        errno = *(boost::get_error_info<skylark::base::error_code>(ex));
        std::cout << "Caught exception, exiting with error " << errno << ": ";
        std::cout << skylark_strerror(errno) << std::endl;
        BOOST_FAIL("Exception when reading arc list.");
    }

    std::stringstream ss;
    ss << "Read matrix (" << A.local_height() << " x " << A.local_width()
       << ") on " << world.rank() << "." << std::endl;
    std::cout << ss.str() << std::flush;

    world.barrier();

    // holds for vc/star (some rows can be shorter if the last element was
    // not set)
    BOOST_REQUIRE(A.local_width() <= A.width());

    El::Int total_rows = 0;
    boost::mpi::all_reduce(
        world, A.local_height(), total_rows, std::plus<El::Int>());
    BOOST_REQUIRE(total_rows == A.height());

    // check the values/coords
    skylark::base::sparse_matrix_t<double> X;
    try {
        std::string fname(argv[1]);
        skylark::utility::io::ReadArcList(fname, X, world, true);
    } catch (skylark::base::skylark_exception ex) {
        SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
        SKYLARK_PRINT_EXCEPTION_TRACE(ex);
        errno = *(boost::get_error_info<skylark::base::error_code>(ex));
        std::cout << "Caught exception, exiting with error " << errno << ": ";
        std::cout << skylark_strerror(errno) << std::endl;
        BOOST_FAIL("Exception when reading arc list (serial).");
    }

    typedef skylark::base::sparse_matrix_t<double>::index_type index_type;

    const index_type* ref_indptr  = X.indptr();
    const index_type* ref_indices = X.indices();
    const double* ref_values      = X.locked_values();

    const index_type* indptr  = A.indptr();
    const index_type* indices = A.indices();
    const double* values      = A.locked_values();

    // traverse local data and compare
    for (int col = 0; col < A.local_width(); col++) {
        int k = ref_indptr[col];
        for (int j = indptr[col]; j < indptr[col + 1]; j++) {
            int row = A.global_row(indices[j]);
            while(ref_indices[k] != row) k++;
            BOOST_REQUIRE(values[j] == ref_values[k]);
        }
    }

    return 0;
}
