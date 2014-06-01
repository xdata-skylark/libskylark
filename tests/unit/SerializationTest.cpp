#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

//#include "../../sketch/CT.hpp"
//#include "../../sketch/CWT.hpp"
#include "../../sketch/sketch.hpp"
#include "../../base/context.hpp"

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

    namespace mpi = boost::mpi;
    mpi::environment env(argc, argv);
    mpi::communicator world;
    const size_t rank = world.rank();
    skylark::base::context_t context (seed);

    double count = 1.0;

    const size_t matrix_full = n * m;
    mpi_vector_t colsf(matrix_full);
    mpi_vector_t rowsf(matrix_full);
    mpi_vector_t valsf(matrix_full);

    for(size_t i = 0; i < matrix_full; ++i) {
        colsf.SetElement(i, i % m);
        rowsf.SetElement(i, i / m);
        valsf.SetElement(i, count);
        count++;
    }

    DistMatrixType A(n, m, rowsf, colsf, valsf);

    //////////////////////////////////////////////////////////////////////////
    //[> Setup test <]

    //[> 1. Create the sketching matrix and dump JSON <]
    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType>
        Sparse(n, n_s, context);

    // dump to property tree
    boost::property_tree::ptree pt = Sparse.get_data()->to_ptree();

    //[> 2. Dump the JSON string to file <]
    std::ofstream out("sketch.json");
    write_json(out, pt);
    out.close();

    //[> 3. Create a sketch from the JSON file. <]
    std::ifstream file;
    std::stringstream json;
    file.open("sketch.json", std::ios::in);

    boost::property_tree::ptree json_tree;
    boost::property_tree::read_json(file, json_tree);

    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType> tmp(json_tree);

    //[> 4. Both sketches should compute the same result. <]
    mpi_vector_t zero;
    DistMatrixType sketch_A(n_s, m, zero, zero, zero);
    DistMatrixType sketch_Atmp(n_s, m, zero, zero, zero);

    Sparse.apply(A, sketch_A, skylark::sketch::columnwise_tag());
    tmp.apply(A, sketch_Atmp, skylark::sketch::columnwise_tag());

    if (!static_cast<bool>(sketch_A == sketch_Atmp))
        BOOST_FAIL("Applied sketch did not result in same result");

    return 0;
}
