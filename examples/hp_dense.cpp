////////////////////////////////////////////////////////////////////////////////

/** To enforce matrix output from root process only */
// #define ROOT_OUTPUT

/** To enforce matrix generation for local->local */
// #define LOCAL

/** To denote whether rowwise sketching is attempted (columnwise otherwise) */
#define ROWWISE


////////////////////////////////////////////////////////////////////////////////

#include <boost/mpi.hpp>
#include <El.hpp>
#include <skylark.hpp>
#include <iostream>


/** Aliases */

typedef El::Matrix<double>     dense_matrix_t;

typedef El::DistMatrix<double> dist_dense_matrix_t;
typedef El::DistMatrix<double, El::VC, El::STAR>
dist_VC_STAR_dense_matrix_t;
typedef El::DistMatrix<double, El::VR, El::STAR>
dist_VR_STAR_dense_matrix_t;
typedef El::DistMatrix<double, El::STAR, El::VC>
dist_STAR_VC_dense_matrix_t;
typedef El::DistMatrix<double, El::STAR, El::VR>
dist_STAR_VR_dense_matrix_t;

typedef El::DistMatrix<double, El::CIRC, El::CIRC>
dist_CIRC_CIRC_dense_matrix_t;

typedef El::DistMatrix<double, El::STAR, El::STAR>
dist_STAR_STAR_dense_matrix_t;


/* Set the following 2 typedefs for various matrix-type tests */
typedef dist_dense_matrix_t input_matrix_t;
typedef dist_dense_matrix_t output_matrix_t;

typedef skylark::sketch::JLT_t<input_matrix_t, output_matrix_t>
sketch_transform_t;


int main(int argc, char* argv[]) {

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    /** Initialize Elemental */
    El::Initialize (argc, argv);

    MPI_Comm mpi_world(world);
    El::Grid grid(mpi_world);

    /** Example parameters */
    int height      = 20;
    int width       = 10;
    int sketch_size = 5;

    /** Define input matrix A */

#ifdef LOCAL
    input_matrix_t A;
    El::Uniform(A, height, width);
#else
    dist_CIRC_CIRC_dense_matrix_t A_CIRC_CIRC(grid);
    input_matrix_t A(grid);
    El::Uniform(A_CIRC_CIRC, height, width);
    A = A_CIRC_CIRC;
#endif

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

#ifdef ROOT_OUTPUT
    if (world.rank() == 0) {
#endif
        El::Print(sketched_A, "sketched_A");
#ifdef ROOT_OUTPUT
    }
#endif
    El::Finalize();
    return 0;
}
