#include <elemental.hpp>
#include <skylark.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include "hilbert.hpp"

namespace bmpi =  boost::mpi;
namespace po = boost::program_options;
using namespace std;

int main (int argc, char** argv) {
	/* Initialize MPI */

	  hilbert_options_t options (argc, argv);
	  if (options.exit_on_return) { return -1; }



	  bmpi::environment env (argc, argv);

	/* Create a global communicator */
	  bmpi::communicator world;


	 skylark::sketch::context_t context (12345, world);

	 elem::Initialize (argc, argv);
	 MPI_Comm mpi_world(world);
	 //elem::Grid grid (mpi_world);

	 DistInputMatrixType X, Y;

	 if (context.rank==0)
		 std::cout << options.print();

	 read_libsvm_dense(context, options.trainfile, X, Y);

	 lossfunction *loss = NULL;
	 switch(options.lossfunction) {
	 	 case SQUARED:
	 		 loss = new squaredloss();
	 		 break;
	 	 case HINGE:
	 		 loss = new hingeloss();
	 		 break;
	 }

	 regularization *regularizer = NULL;
	 switch(options.regularizer) {
	 	 case L2:
	 		 regularizer = new l2();
	 		 break;
	 }

	 FeatureTransform *featureMap = NULL;
	 switch(options.kernel) {
	 	 case LINEAR:
	 		 featureMap = new Identity();
	 		 options.randomfeatures = X.Width();
	 		 break;
	 }


	 // int k = Y.Width();
	 int k;
	 int kmax = *std::max_element(Y.Buffer(), Y.Buffer() + Y.LocalHeight());

	 boost::mpi::all_reduce(context.comm, kmax, k, boost::mpi::maximum<int>());
	 if (k>1) // we assume 0-to-N encoding of classes. Hence N = k+1. For two classes, k=1.
	 	k++;

	 BlockADMMSolver *Solver = new BlockADMMSolver(
			 	 	 loss,
	 				 regularizer,
	 				 featureMap,
	 				 options.lambda,
	 				 options.randomfeatures,
	 				 options.numfeaturepartitions,
	 				 options.numthreads,
	 				 options.tolerance,
	 				 options.MAXITER,
	 				 options.rho);

	 elem::Matrix<double> Wbar(options.randomfeatures, k);
	 elem::MakeZeros(Wbar);
	 Solver->train(context, X,Y,Wbar);
	 if (context.rank==0) {
		 std::stringstream dimensionstring;
		 dimensionstring << "# Dimensions " << options.randomfeatures << " " << k << "\n";
		 elem::Write(Wbar, options.print().append(dimensionstring.str()), options.modelfile);
	 }
//	 cout << " Rank " << context.rank << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << endl;

	 elem::Finalize();
	 return 0;
}
