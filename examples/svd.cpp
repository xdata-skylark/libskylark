#include <iostream>
#include <functional>
#include <cstring>
#include <vector>
#include <utility>
#include <ext/hash_map>

#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <skylark.hpp>

#include "utilities.hpp"
#include "parser.hpp"

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skyb =  skylark::base;
/*******************************************/

/* These were declared as extern in utilities.hpp --- defining it here */
int int_params [NUM_INT_PARAMETERS];
char* chr_params[NUM_CHR_PARAMETERS];

/** Typedef DistMatrix and Matrix */
typedef std::vector<int> IntContainer;
typedef std::vector<double> DblContainer;
typedef elem::Matrix<double> MatrixType;
typedef elem::DistMatrix<double, elem::VR, elem::STAR> DistMatrixType;

int main (int argc, char** argv) {
    /* Initialize Elemental (and MPI) */
    elem::Initialize (argc, argv);

    /* MPI sends argc and argv everywhere --- parse everywhere */
    parse_parameters (argc,argv);

    // get communicator
    boost::mpi::communicator comm;
    int rank = comm.rank();

    MPI_Comm mpi_world(world);
    elem::Grid grid (mpi_world);

    /* Initialize skylark */
    skyb::context_t context (int_params[RAND_SEED_INDEX], comm);

    int m = int_params[M_INDEX];
    int n = int_params[N_INDEX];
    int k = int_params[S_INDEX];

    /* Create matrices A and B */
    DistMatrixType A1(grid);
    DistMatrixType A2(grid);

    elem::DistMatrix<double> A11,A22,A33,V33;
    DistMatrixType s33(grid);
    DistMatrixType A(grid);
    DistMatrixType U(grid);
    MatrixType s(k,1);
    MatrixType V(n,k);


    /** Only randomization is supported for now */
    if (0==int_params[USE_RANDOM_INDEX]) {
        /** TODO: Read the entries! */
        std::cout << "We don't support reading --- yet --" << std::endl;
    } else {
        elem::Uniform (A1, m, k);
        elem::Uniform (A2, m, k);
        elem::Matrix<double> A3(k,k);

        skylark::nla::Gemm(A1,A2,A3,context);

        elem::Uniform (A3, k, k);

        skylark::nla::Gemm(A1,A3,A2);

        elem::Uniform(A11, m, k);
        elem::Uniform(A22, k, n);
        A33.Resize(m,n);
        Gemm(elem::NORMAL, elem::NORMAL, 1.0, A11, A22, 0.0, A33);
        A.Resize(m,n);
        A = A33;
        elem::SVD(A33, s33, V33);
        elem::Display(s33, "True singular values");

    }

    /**
     * Depending on which sketch is requested, do the sketching.
     */
    if (0==strcmp("JLT", chr_params[TRANSFORM_INDEX]) ) {

        if (SKETCH_LEFT == int_params[SKETCH_DIRECTION_INDEX]) {
            elem::Display(s, "Approximate singular values");
        } else {
            //std::cout << "We only have left sketching. Please retry" << std::endl;
        }
    } else {
        std::cout << "We only have JLT sketching. Please retry" <<
            std::endl;
    }

    elem::Finalize();

    return 0;
}
