/**
 *  This test ensures that our internal distributed sparse matrix behave as
 *  their Elemental counter parts.
 *  This test builds on the following assumptions:
 *
 *      - Elemental distributions and Gemms are implemented correctly.
 */


#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#define SKYLARK_NO_ANY
#include "../../base/sparse_vc_star_matrix.hpp"
#include "../../base/sparse_star_vr_matrix.hpp"

typedef El::DistMatrix<double, El::VC, El::STAR> dense_vc_star_matrix_t;
typedef skylark::base::sparse_vc_star_matrix_t<double> sparse_vc_star_matrix_t;

typedef El::DistMatrix<double, El::STAR, El::VR> dense_star_vr_matrix_t;
typedef skylark::base::sparse_star_vr_matrix_t<double> sparse_star_vr_matrix_t;


template <typename dense_matrix_t, typename sparse_matrix_t>
void test_matrix_properties(
        const dense_matrix_t& A, const sparse_matrix_t& A_sparse) {

    //FIXME: test more

    BOOST_REQUIRE(A_sparse.width()    == A.Width());
    BOOST_REQUIRE(A_sparse.height()   == A.Height());
    BOOST_REQUIRE(A_sparse.nonzeros() == A.Width() * A.Height());

    BOOST_REQUIRE(A_sparse.local_width()  == A.LocalWidth());
    BOOST_REQUIRE(A_sparse.local_height() == A.LocalHeight());
}


template <typename dense_matrix_t, typename sparse_matrix_t>
void compare_matrix_values(
        const dense_matrix_t& A, const sparse_matrix_t& A_sparse) {

    const int* indptr    = A_sparse.indptr();
    const int* indices   = A_sparse.indices();
    const double* values = A_sparse.locked_values();

    for(int col = 0; col < A_sparse.local_width(); col++) {
        for(int idx = indptr[col]; idx < indptr[col + 1]; idx++) {

            int row = indices[idx];
            BOOST_REQUIRE(A.GetLocal(row, col) == values[idx]);
        }
    }
}


int test_main(int argc, char *argv[]) {

    /** Initialize Elemental */
    El::Initialize (argc, argv);

    /** Initialize MPI  */
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    MPI_Comm mpi_world(world);
    El::Grid grid(mpi_world);


    //////////////////////////////////////////////////////////////////////////
    //[> Parameters <]

    //FIXME: use random sizes?
    const int height = 20;
    const int width  = 10;

    //////////////////////////////////////////////////////////////////////////
    //[> Setup test <]

    dense_vc_star_matrix_t A_vc(grid);
    El::Uniform(A_vc, height, width);
    El::Zero(A_vc);

    sparse_vc_star_matrix_t A_sparse_vc(height, width, world, grid);

    double count = 0.0;
    for(int col = 0; col < A_vc.Width(); col++) {
        for(int row = 0; row < A_vc.Height(); row++) {
            A_vc.Update(row, col, count);
            A_sparse_vc.queue_update(row, col, count);
            count++;
        }
    }

    A_sparse_vc.finalize();


    dense_star_vr_matrix_t A_vr(grid);
    El::Uniform(A_vr, height, width);
    El::Zero(A_vr);

    sparse_star_vr_matrix_t A_sparse_vr(height, width, world, grid);

    count = 0.0;
    for(int col = 0; col < A_vr.Width(); col++) {
        for(int row = 0; row < A_vr.Height(); row++) {
            A_vr.Update(row, col, count);
            A_sparse_vr.queue_update(row, col, count);
            count++;
        }
    }

    A_sparse_vr.finalize();


    //////////////////////////////////////////////////////////////////////////
    //[> Test properties <]

    test_matrix_properties(A_vc, A_sparse_vc);
    test_matrix_properties(A_vr, A_sparse_vr);

    compare_matrix_values(A_vc, A_sparse_vc);
    compare_matrix_values(A_vr, A_sparse_vr);

    El::Finalize();
    return 0;
}
