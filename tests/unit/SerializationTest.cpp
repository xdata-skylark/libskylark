#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "../../sketch/CT.hpp"
#include "../../sketch/CWT.hpp"
#include "../../base/context.hpp"
#include "../../utility/sketch_archive.hpp"

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
    //FIXME: improve interface (remove indirection)
    boost::property_tree::ptree pt;
    pt << Sparse;
    skylark::utility::sketch_archive_t ar;
    ar << pt;

    //[> 2. Dump the JSON string to file <]
    std::ofstream out("sketch.json");
    out << ar;
    out.close();

    //[> 3. Create a sketch from the JSON file. <]
    std::ifstream file;
    std::stringstream json;
    file.open("sketch.json", std::ios::in);
    skylark::utility::sketch_archive_t arl;
    file >> arl;
    skylark::sketch::CWT_t<DistMatrixType, DistMatrixType> tmp(arl.get(0));

    mpi_vector_t zero;
    DistMatrixType sketch_A(n_s, m, zero, zero, zero);
    DistMatrixType sketch_Atmp(n_s, m, zero, zero, zero);

    Sparse.apply(A, sketch_A, skylark::sketch::columnwise_tag());
    tmp.apply(A, sketch_Atmp, skylark::sketch::columnwise_tag());

    if (!static_cast<bool>(sketch_A == sketch_Atmp))
        BOOST_FAIL("Applied sketch did not result in same result");


    //[> 4. Serialize two sketches in one file <]
    typedef elem::DistMatrix<double, elem::VR, elem::STAR> DenseDistMat_t;
    elem::Initialize (argc, argv);
    elem::Grid grid (world);

    //DenseDistMat_t A(grid);
    //elem::Uniform(A, m, n);

    skylark::sketch::CT_t<DenseDistMat_t, DenseDistMat_t>
        Dense(n, n_s, 2.2, context);
    boost::property_tree::ptree ptd;
    ptd << Dense;

    skylark::utility::sketch_archive_t ar2;
    ar2 << ptd << pt;

    //TODO: check if as expected
    std::cout << ar2 << std::endl;

    return 0;
}
