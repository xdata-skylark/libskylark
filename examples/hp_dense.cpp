/** Selecting the parts of HP support to activate */
////////////////////////////////////////////////////////////////////////////////

// #define ROWWISE

#define HP_DENSE_TRANSFORM_ELEMENTAL
#define HP_DENSE_TRANSFORM_ELEMENTAL_MC_MR
#define OPTIMIZED

#define TESTED_ROWWISE    inner_panel_gemm(A, sketch_of_A, tag); return;
#define TESTED_COLUMNWISE panel_matrix_gemm(A, sketch_of_A, tag); return;

////////////////////////////////////////////////////////////////////////////////


#include <boost/mpi.hpp>
#include <elemental.hpp>
#include <skylark.hpp>

#include <iostream>


/** Temporarily copied utilities */
template<typename MatrixType>
MatrixType operator-(MatrixType& A, MatrixType& B) {
    MatrixType C;
    elem::Copy(A, C);
    elem::Axpy(-1.0, B, C);
    return C;
}

template<typename MatrixType>
double diff_norm(MatrixType& A, MatrixType& B) {
    MatrixType C = A - B;
    double diff_norm = elem::Norm(C);
    return diff_norm;
}


template<typename MatrixType>
bool equal(MatrixType& A, MatrixType& B,  double threshold=1.e-4) {
    MatrixType C = A - B;
    double diff_norm = elem::Norm(C);
    if (diff_norm < threshold) {
        return true;
    }
    return false;
}



/** Aliases */

typedef elem::Matrix<double>     matrix_t;
typedef elem::DistMatrix<double> dist_matrix_t;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR>
 dist_STAR_STAR_matrix_t;
typedef skylark::sketch::JLT_t<matrix_t, matrix_t> ref_sketch_transform_t;
typedef skylark::sketch::JLT_t<dist_matrix_t, dist_matrix_t> sketch_transform_t;

int main(int argc, char* argv[]) {

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    /** Initialize Elemental */
    elem::Initialize (argc, argv);

    MPI_Comm mpi_world(world);
    elem::Grid grid(mpi_world);

    /** Example parameters */
    int height = 200;
    int width  = 100;
    int sketch_size = 50;

    /** Define input matrix A */
    dist_STAR_STAR_matrix_t A_STAR_STAR(grid);
    dist_matrix_t A(grid);
    elem::Uniform(A_STAR_STAR, height, width);
    A = A_STAR_STAR;

    /** Initialize contexts */
    skylark::base::context_t ref_context(0);
    skylark::base::context_t context(0);

#ifdef ROWWISE
    /** Sketch transform (rowwise)*/
    int size = width;

    /** Local matrix computation at root */
    dist_STAR_STAR_matrix_t ref_sketched_A_STAR_STAR(height, sketch_size);
    dist_STAR_STAR_matrix_t     sketched_A_STAR_STAR(grid);

    if (world.rank() == 0) {
        const matrix_t A_local = A_STAR_STAR.Matrix();
        matrix_t sketched_A_local = ref_sketched_A_STAR_STAR.Matrix();
        ref_sketch_transform_t ref_sketch_transform(size,
            sketch_size,
            ref_context);
        ref_sketch_transform.apply(A_local,
            sketched_A_local,
            skylark::sketch::rowwise_tag());
    }

    /** Distributed matrix computation */
    dist_matrix_t sketched_A(height, sketch_size);
    sketch_transform_t sketch_transform(size, sketch_size, context);
    sketch_transform.apply(A, sketched_A, skylark::sketch::rowwise_tag());
    sketched_A_STAR_STAR = sketched_A;


#else

    /** Sketch transform (columnwise)*/
    int size = height;
    dist_matrix_t sketched_A(sketch_size, width);
    sketch_transform_t sketch_transform(size, sketch_size, context);
    sketch_transform.apply(A, sketched_A, skylark::sketch::columnwise_tag());


#endif

    /** Print sketched_A */
    //std::cout << diff_norm(ref_sketched_A_STAR_STAR, sketched_A_STAR_STAR)
    //          << std::endl;
    elem::Print(sketched_A, "sketched_A");

    elem::Finalize();
    return 0;
}
