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
namespace skys =  skylark::sketch;
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
    /* Initialize MPI */
    bmpi::environment env (argc, argv);

    /* Create a global communicator */
    bmpi::communicator world;

    /* MPI sends argc and argv everywhere --- parse everywhere */
    parse_parameters (argc,argv);

    /* Initialize elemental */
    elem::Initialize (argc, argv);
    MPI_Comm mpi_world(world);
    elem::Grid grid (mpi_world);

    /* Initialize skylark */
    skys::context_t context (int_params[RAND_SEED_INDEX], world);

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
    //DistMatrixType U(m,k);
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
      //  A1.Print("A1");
      //  A2.Print("A2");

        skylark::nla::Gemm(A1,A2,A3,context);
        if (context.rank == 0) {
  //      		A3.Print("Checking Matrix multiplication [VR,*]-times-[VR,*] A1'A2 = A3");
        }

        elem::Uniform (A3, k, k);

//        A3.Print("Generated new A3:");
        skylark::nla::Gemm(A1,A3,A2);

    //    A2.Print("Checking Matrix multiplication between [VR,*] A1 and Matrix A3");

        //A3.ResizeTo(m, n);
        //V.ResizeTo(m, k);

        elem::Uniform(A11, m, k);
        elem::Uniform(A22, k, n);
        A33.ResizeTo(m,n);
        Gemm(elem::NORMAL, elem::NORMAL, 1.0, A11, A22, 0.0, A33);
        A.ResizeTo(m,n);
        A = A33;
        elem::SVD(A33, s33, V33);
        elem::Display(s33, "True singular values");

        //A.ResizeTo(m, n);
        //A = A3;
      //  elem::Uniform (int_params[M_INDEX], int_params[N_RHS_INDEX], B);
    }

    /**
     * Depending on which sketch is requested, do the sketching.
     */
    if (0==strcmp("JLT", chr_params[TRANSFORM_INDEX]) ) {

        if (SKETCH_LEFT == int_params[SKETCH_DIRECTION_INDEX]) {
            //	A.Print("Matrix A");
            //U.ResizeTo(m,k);
            // skylark::nla::SVD(A, U, s, V, int_params[S_INDEX], 1, context);
            //	U.Print("U: Left Singular Vectors");
            elem::Display(s, "Approximate singular values");
            //	V.Print("V: Right Singular Vectors");
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
