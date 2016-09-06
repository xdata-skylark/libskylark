#include <boost/mpi.hpp>
#include <El.hpp>
#include <iostream>
#include <boost/test/minimal.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

#include "test_utils.hpp"


/** Aliases */

typedef El::Matrix<double>     dense_matrix_t;
typedef El::DistMatrix<double> dist_dense_matrix_t;
typedef El::DistMatrix<double, El::CIRC, El::CIRC>
dist_CIRC_CIRC_dense_matrix_t;

typedef dist_dense_matrix_t input_matrix_t;
typedef dist_dense_matrix_t output_matrix_t;
typedef skylark::sketch::JLT_t<input_matrix_t, output_matrix_t>
sketch_transform_t;

typedef skylark::sketch::JLT_t<dense_matrix_t, dense_matrix_t>
sketch_transform_local_t;


int test_main(int argc, char* argv[]) {

    /** Initialize Elemental */
    El::Initialize (argc, argv);

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    MPI_Comm mpi_world(world);
    El::Grid grid(mpi_world);

    /** Example parameters */
    int height      = 20;
    int width       = 10;
    int sketch_size = 5;

    dist_CIRC_CIRC_dense_matrix_t A_CIRC_CIRC(grid);
    input_matrix_t A(grid);
    El::Uniform(A_CIRC_CIRC, height, width);
    A = A_CIRC_CIRC;
    int size;


    /** Sketch transform rowwise (rw) */
    size = width;
    skylark::base::context_t context_rw(0);
    output_matrix_t sketched_A_rw(height, sketch_size, grid);
    sketch_transform_t sketch_transform_rw(size, sketch_size, context_rw);
    sketch_transform_rw.apply(A, sketched_A_rw,
        skylark::sketch::rowwise_tag());
    dist_CIRC_CIRC_dense_matrix_t sketched_A_rw_CIRC_CIRC = sketched_A_rw;

    if(world.rank() == 0) {
        dense_matrix_t sketched_A_rw_gathered =
            sketched_A_rw_CIRC_CIRC.Matrix();

        skylark::base::context_t context_rw_local(0);
        dense_matrix_t A_rw_local = A_CIRC_CIRC.Matrix();
        dense_matrix_t sketched_A_rw_local(height, sketch_size);
        sketch_transform_local_t sketch_transform_rw_local(size, sketch_size,
            context_rw_local);
        sketch_transform_rw_local.apply(A_rw_local, sketched_A_rw_local,
            skylark::sketch::rowwise_tag());

        if (!test::util::equal(sketched_A_rw_gathered, sketched_A_rw_local))
            BOOST_FAIL("Rowwise sketching resuts are not equal");
    }


    /** Sketch transform columnwise (cw) */
    size = height;
    skylark::base::context_t context_cw(0);
    output_matrix_t sketched_A_cw(sketch_size, width, grid);
    sketch_transform_t sketch_transform_cw(size, sketch_size, context_cw);
    sketch_transform_cw.apply(A, sketched_A_cw,
        skylark::sketch::columnwise_tag());
    dist_CIRC_CIRC_dense_matrix_t sketched_A_cw_CIRC_CIRC = sketched_A_cw;

    if(world.rank() == 0) {
        dense_matrix_t sketched_A_cw_gathered =
            sketched_A_cw_CIRC_CIRC.Matrix();

        skylark::base::context_t context_cw_local(0);
        dense_matrix_t A_cw_local = A_CIRC_CIRC.Matrix();
        dense_matrix_t sketched_A_cw_local(sketch_size, width);
        sketch_transform_local_t sketch_transform_cw_local(size, sketch_size,
            context_cw_local);
        sketch_transform_cw_local.apply(A_cw_local, sketched_A_cw_local,
            skylark::sketch::columnwise_tag());

        if (!test::util::equal(sketched_A_cw_gathered, sketched_A_cw_local))
            BOOST_FAIL("Columnwise sketching resuts are not equal");
    }


    El::Finalize();
    return 0;
}
