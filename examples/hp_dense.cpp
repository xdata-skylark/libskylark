/** Selecting the parts of HP support to activate */
// #define HP_DENSE_TRANSFORM_ELEMENTAL
// #define HP_DENSE_TRANSFORM_ELEMENTAL_MC_MR

#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <skylark.hpp>

#include <iostream>

/** Aliases */
typedef elem::DistMatrix<double> dist_matrix_t;
typedef skylark::sketch::JLT_t<dist_matrix_t, dist_matrix_t> sketch_transform_t;

int main(int argc, char* argv[]) {

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    MPI_Comm mpi_world(world);
    elem::Grid grid(mpi_world);

    /** Initialize Elemental */
    elem::Initialize (argc, argv);

    /** Example parameters */
    int height = 500;
    int width  = 300;
    int sketch_size = 100;

    /** Define input matrix A */
    dist_matrix_t A(grid);
    elem::Uniform(A, height, width);

    /** Initialize context */
    skylark::base::context_t context(0);

#if 1
    /** Sketch transform (rowwise)*/
    int size = width;
    dist_matrix_t sketched_A(size, sketch_size);
    sketch_transform_t sketch_transform(size, sketch_size, context);
    sketch_transform.apply(A, sketched_A, skylark::sketch::rowwise_tag());
#endif

#if 0
    /** Sketch transform (columnwise)*/
    int size = height;
    dist_matrix_t sketched_A(sketch_size, size);
    sketch_transform_t sketch_transform(size, sketch_size, context);
    sketch_transform.apply(A, sketched_A, skylark::sketch::columnwise_tag());
#endif

    /** Print sketched_A */
    elem::Print(sketched_A, "sketched_A");

    elem::Finalize();
    return 0;
}
