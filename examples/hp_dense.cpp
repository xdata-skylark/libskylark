/** Selecting the parts of HP support to activate */
////////////////////////////////////////////////////////////////////////////////

// #define ROWWISE

#define OPTIMIZED

#define TESTED_ROWWISE    inner_panel_gemm(A, sketch_of_A, tag); return;
#define TESTED_COLUMNWISE inner_panel_gemm(A, sketch_of_A, tag); return;

#define HP_DENSE_TRANSFORM_ELEMENTAL
#define HP_DENSE_TRANSFORM_ELEMENTAL_MC_MR
#define HP_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR
#define HP_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST
#define HP_DENSE_TRANSFORM_ELEMENTAL_MC_MR_LOCAL
#define HP_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_LOCAL
#define HP_DENSE_TRANSFORM_ELEMENTAL_STAR_ROWDIST_LOCAL

////////////////////////////////////////////////////////////////////////////////


#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <skylark.hpp>
#include <iostream>


/** Aliases */

typedef elem::Matrix<double>     dense_matrix_t;

typedef elem::DistMatrix<double> dist_dense_matrix_t;
typedef elem::DistMatrix<double, elem::VC, elem::STAR>
dist_VC_STAR_dense_matrix_t;
typedef elem::DistMatrix<double, elem::VR, elem::STAR>
dist_VR_STAR_dense_matrix_t;
typedef elem::DistMatrix<double, elem::STAR, elem::VC>
dist_STAR_VC_dense_matrix_t;
typedef elem::DistMatrix<double, elem::STAR, elem::VR>
dist_STAR_VR_dense_matrix_t;

typedef elem::DistMatrix<double, elem::CIRC, elem::CIRC>
dist_CIRC_CIRC_dense_matrix_t;

/* Set the following 2 typedefs for various matrix-type tests */
typedef dist_VC_STAR_dense_matrix_t input_matrix_t;
typedef dist_VC_STAR_dense_matrix_t output_matrix_t;


typedef skylark::sketch::JLT_t<input_matrix_t, output_matrix_t>
sketch_transform_t;


int main(int argc, char* argv[]) {

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    /** Initialize Elemental */
    elem::Initialize (argc, argv);

    MPI_Comm mpi_world(world);
    elem::Grid grid(mpi_world);

    /** Example parameters */
    int height      = 20;
    int width       = 10;
    int sketch_size = 5;

    /** Define input matrix A */
    dist_CIRC_CIRC_dense_matrix_t A_CIRC_CIRC(grid);
    input_matrix_t A(grid);
    elem::Uniform(A_CIRC_CIRC, height, width);
    A = A_CIRC_CIRC;

    /** Initialize context */
    skylark::base::context_t context(0);

#ifdef ROWWISE

    /** Sketch transform (rowwise)*/
    int size = width;
    /** Distributed matrix computation */
    output_matrix_t sketched_A(height, sketch_size);
    sketch_transform_t sketch_transform(size, sketch_size, context);
    sketch_transform.apply(A, sketched_A, skylark::sketch::rowwise_tag());

#else

    /** Sketch transform (columnwise)*/
    int size = height;
    /** Distributed matrix computation */
    output_matrix_t sketched_A(sketch_size, width);
    sketch_transform_t sketch_transform(size, sketch_size, context);
    sketch_transform.apply(A, sketched_A, skylark::sketch::columnwise_tag());

#endif

    elem::Print(sketched_A, "sketched_A");

    elem::Finalize();
    return 0;
}
