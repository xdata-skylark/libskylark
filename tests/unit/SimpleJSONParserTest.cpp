#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#include "../../utility/distributions.hpp"
#include "../../utility/simple_json_parser.hpp"

#include "../../sketch/context.hpp"
#include "../../sketch/hash_transform.hpp"
#include "../../sketch/CWT.hpp"


template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct Dummy_t : public skylark::sketch::hash_transform_t<
    InputMatrixType, OutputMatrixType,
    boost::random::uniform_int_distribution,
    skylark::utility::rademacher_distribution_t > {

    typedef skylark::sketch::hash_transform_t<
        InputMatrixType, OutputMatrixType,
        boost::random::uniform_int_distribution,
        skylark::utility::rademacher_distribution_t >
            hash_t;

    Dummy_t(int N, int S, skylark::sketch::context_t& context)
        : skylark::sketch::hash_transform_t<InputMatrixType, OutputMatrixType,
          boost::random::uniform_int_distribution,
          skylark::utility::rademacher_distribution_t>(N, S, context)
    {}

    Dummy_t(const std::string json_filename,  skylark::sketch::context_t& context)
        : skylark::sketch::hash_transform_t<InputMatrixType, OutputMatrixType,
          boost::random::uniform_int_distribution,
          skylark::utility::rademacher_distribution_t>(json_filename, context)
    {}

    size_t rowidx(size_t i) { return hash_t::row_idx[i]; }
    double rowval(size_t i) { return hash_t::row_value[i]; }
};

int test_main(int argc, char *argv[]) {

    //////////////////////////////////////////////////////////////////////////
    //[> Parameters <]

    const size_t n   = 10;
    const size_t m   = 5;
    const size_t n_s = 6;
    const size_t m_s = 3;

    const int seed = static_cast<int>(rand() * 100);

    typedef FullyDistVec<size_t, double> mpi_vector_t;
    typedef SpDCCols<size_t, double> col_t;
    typedef SpParMat<size_t, double, col_t> DistMatrixType;

    //////////////////////////////////////////////////////////////////////////
    //[> Setup test <]
    namespace mpi = boost::mpi;
    mpi::environment env(argc, argv);
    mpi::communicator world;
    const size_t rank = world.rank();
    skylark::sketch::context_t context (seed, world);

    //[> 1. Create the sketching matrix and dump JSON <]
    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType> Sparse(n, n_s, context);

    std::string json_object;
    Sparse.dump_json(json_object);

    //[> 2. Dump the JSON string to file <]
    std::cout << json_object << std::endl;
    std::ofstream out("sketch.json");
    out << json_object;
    out.close();

    //[> 3. Create a new context and sketch from the JSON file. <]
    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType>
        Sparse_load("sketch.json", context);

    //for(size_t i = 0; i < n; ++i)
        //BOOST_ASSERT( (Sparse.rowidx(i) == Sparse_cl.rowidx(i)) &&
                      //(Sparse.rowval(i) == Sparse_cl.rowval(i)));



    //if (!static_cast<bool>(expected_A == sketch_A))
        //BOOST_FAIL("Result of colwise application not as expected");

    return 0;
}
