#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "../../utility/distributions.hpp"
#include "../../utility/sketch_archive.hpp"

#include "../../sketch/context.hpp"
#include "../../sketch/transform_data.hpp"
#include "../../sketch/hash_transform.hpp"
#include "../../sketch/CWT.hpp"
#include "../../sketch/CT.hpp"

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

    // dump to property tree
    //FIXME: improve interface (remove indirection)
    boost::property_tree::ptree pt;
    pt << Sparse;
    skylark::utility::sketch_archive_t ar;
    ar << pt;

    //[> 2. Dump the JSON string to file <]
    std::ofstream out("sketch.json");
    out << ar;
    out.close();

    //[> 3. Create a new context and sketch from the JSON file. <]
    std::ifstream file;
    std::stringstream json;
    file.open("sketch.json", std::ios::in);
    skylark::utility::sketch_archive_t arl;
    file >> arl;
    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType> tmp(
            arl.get(0), context);

    //for(size_t i = 0; i < n; ++i)
        //BOOST_ASSERT( (Sparse.rowidx(i) == Sparse_cl.rowidx(i)) &&
                      //(Sparse.rowval(i) == Sparse_cl.rowval(i)));



    //[> 4. Serialize two sketches in one file <]
    typedef elem::DistMatrix<double, elem::VR, elem::STAR> DenseDistMat_t;
    elem::Initialize (argc, argv);
    elem::Grid grid (world);

    DenseDistMat_t A(grid);
    elem::Uniform(A, m, n);

    skylark::sketch::CT_t<DenseDistMat_t, DenseDistMat_t> Dense(n, n_s, 2.2, context);
    boost::property_tree::ptree ptd;
    ptd << Dense;

    skylark::utility::sketch_archive_t ar2;
    ar2 << ptd << pt;

    std::cout << ar2 << std::endl;

    return 0;
}
