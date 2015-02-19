#include <iostream>
#include <functional>
#include <cstring>
#include <vector>
#include <utility>
#include <ext/hash_map>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <skylark.hpp>

#include "utilities.hpp"
#include "parser.hpp"

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skys =  skylark::sketch;
namespace skyb =  skylark::base;
/*******************************************/

/* These were declared as extern in utilities.hpp --- defining it here */
int int_params [NUM_INT_PARAMETERS];
char* chr_params[NUM_CHR_PARAMETERS];

/** Typedef DistMatrix and Matrix */
typedef std::vector<int> IntContainer;
typedef std::vector<double> DblContainer;
typedef El::DistMatrix<double, El::CIRC, El::CIRC> MatrixType;
typedef El::DistMatrix<double, El::VC, El::STAR> DistMatrixType;

int main (int argc, char** argv) {
    /* Initialize Elemental */
    El::Initialize (argc, argv);

    /* MPI sends argc and argv everywhere --- parse everywhere */
    parse_parameters (argc,argv);

    /* Initialize skylark */
    skyb::context_t context (int_params[RAND_SEED_INDEX]);

    /* Create matrices A and B */
    bmpi::communicator world;
    MPI_Comm mpi_world(world);
    El::Grid grid (mpi_world);
    El::DistMatrix<double, El::VR, El::STAR> A(grid);
    El::DistMatrix<double, El::VR, El::STAR> B(grid);

    /** Only randomization is supported for now */
    if (0==int_params[USE_RANDOM_INDEX]) {
        /** TODO: Read the entries! */
        std::cout << "We don't support reading --- yet --" << std::endl;
    } else {
        El::Uniform (A, int_params[M_INDEX], int_params[N_INDEX]);
        El::Uniform (B, int_params[M_INDEX], int_params[N_RHS_INDEX]);
    }

    /**
     * Depending on which sketch is requested, do the sketching.
     */
    if (0==strcmp("JLT", chr_params[TRANSFORM_INDEX]) ) {

        if (SKETCH_LEFT == int_params[SKETCH_DIRECTION_INDEX]) {

            /* 1. Create the sketching matrix */
            skys::JLT_t<DistMatrixType, MatrixType> JLT (int_params[M_INDEX],
                int_params[S_INDEX], context);

            /* 2. Create space for the sketched matrix */
            MatrixType sketch_A(int_params[S_INDEX], int_params[N_INDEX]);

            /* 3. Apply the transform */
            try {
                JLT.apply (A, sketch_A, skys::columnwise_tag());
            } catch (skylark::base::skylark_exception ex) {
                SKYLARK_PRINT_EXCEPTION_DETAILS(ex);
                SKYLARK_PRINT_EXCEPTION_TRACE(ex);
                errno = *(boost::get_error_info<skylark::base::error_code>(ex));
                std::cout << "Caught exception, exiting with error " << errno << std::endl;
                std::cout << skylark_strerror(errno) << std::endl;
                return errno;
            }

            /* 4. Print and see the result (if small enough) */
            if (int_params[S_INDEX] * int_params[N_INDEX] < 100 &&
                world.rank() == 0)
                El::Display(sketch_A);

            /** TODO: Do that same to B, and solve the system! */

        } else {
            std::cout << "We only have left sketching. Please retry" << std::endl;
        }
    } else if (0==strcmp("FJLT", chr_params[TRANSFORM_INDEX]) ) {
        if (SKETCH_LEFT == int_params[SKETCH_DIRECTION_INDEX]) {
            /* 1. Create the sketching matrix */
            skys::FJLT_t<DistMatrixType, MatrixType> FJLT (int_params[M_INDEX],
                int_params[S_INDEX], context);

            /* 2. Create space for the sketched matrix */
            MatrixType sketch_A(int_params[S_INDEX], int_params[N_INDEX]);

            /* 3. Apply the transform */
            FJLT.apply (A, sketch_A, skys::columnwise_tag());

            /* 4. Print and see the result */
            if (int_params[S_INDEX] * int_params[M_INDEX] < 100 &&
                world.rank() == 0)
                El::Display(sketch_A);

            /** TODO: Do that same to B, and solve the system! */


        }
    } else if (0==strcmp("CWT", chr_params[TRANSFORM_INDEX]) ) {
        if (SKETCH_LEFT == int_params[SKETCH_DIRECTION_INDEX]) {

            /* 1. Create the sketching matrix */
            skys::CWT_t<DistMatrixType, MatrixType>
                Sparse (int_params[M_INDEX], int_params[S_INDEX], context);

            /* 2. Create space for the sketched matrix */
            MatrixType sketch_A(int_params[S_INDEX], int_params[N_INDEX]);

            /* 3. Apply the transform */
            SKYLARK_BEGIN_TRY()
                Sparse.apply (A, sketch_A, skys::columnwise_tag());
            SKYLARK_END_TRY()
            SKYLARK_CATCH_AND_RETURN_ERROR_CODE();

            /* 4. Print and see the result */
            if (int_params[S_INDEX] * int_params[M_INDEX] < 100 &&
                world.rank() == 0)
                El::Display(sketch_A);

            /** TODO: Do that same to B, and solve the system! */


        }
    } else {
        std::cout << "We only have JLT/FJLT/Sparse sketching. Please retry" <<
            std::endl;
    }

    El::Finalize();

    return 0;

}
