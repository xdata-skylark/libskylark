#include <elemental.hpp>
#include <skylark.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include "kernels.hpp"
#include "hilbert.hpp"
#include <omp.h>


namespace bmpi =  boost::mpi;
namespace po = boost::program_options;
using namespace std;

int main (int argc, char** argv) {
	/* Initialize MPI */

      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	  hilbert_options_t options (argc, argv);
	  if (options.exit_on_return) { return -1; }

	  bmpi::environment env (argc, argv);

	/* Create a global communicator */
	  bmpi::communicator world;


	 skylark::sketch::context_t context (12345, world);

	 elem::Initialize (argc, argv);

	 MPI_Comm mpi_world(world);

	 //elem::Grid grid (mpi_world);

	 omp_set_num_threads(options.numthreads);

	 DistInputMatrixType X, Y;

	 if (context.rank==0)
		 std::cout << options.print();

	 read_libsvm_dense(context, options.trainfile, X, Y);

         cout << " Rank " << context.rank << " on " << env.processor_name() << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << endl;


	 lossfunction *loss = NULL;
	 switch(options.lossfunction) {
	 	 case SQUARED:
	 		 loss = new squaredloss();
	 		 break;
	 	 case HINGE:
	 		 loss = new hingeloss();
	 		 break;
	 	 case LOGISTIC:
	 	     loss = new logisticloss();
	 	     break;
	 }

	 regularization *regularizer = NULL;
	 switch(options.regularizer) {
	 	 case L2:
	 		 regularizer = new l2();
	 		 break;
	 }

	 // int k = Y.Width();
	 int k;
	 int kmax = *std::max_element(Y.Buffer(), Y.Buffer() + Y.LocalHeight());

	 boost::mpi::all_reduce(context.comm, kmax, k, boost::mpi::maximum<int>());
	 if (k>1) // we assume 0-to-N encoding of classes. Hence N = k+1. For two classes, k=1.
	 	k++;

	 BlockADMMSolver *Solver = NULL;
	 int blksize;
	 int features;
	 switch(options.kernel) {
	 	 case LINEAR:
	 		 features = X.Width();
	 		 Solver = new BlockADMMSolver(
	 				context, 
	 				loss,
	 				 regularizer,	
	 				 options.lambda,
	 				 X.Width(),
	 				 options.numfeaturepartitions,
	 				 options.numthreads,
	 				 options.tolerance,
	 				 options.MAXITER,
	 				 options.rho);
	 		 break;
	 		 
	 	 case GAUSSIAN:
	 		 features = options.randomfeatures;
	 		 if (options.regularmap)
		 		 Solver = new BlockADMMSolver(
		 				 context, 
		 				 loss,
		 				 regularizer,	
		 				 options.lambda,
		 				 features,
		 				 skylark::ml::kernels::gaussian_t(X.Width(), options.kernelparam),
		 				 skylark::ml::regular_feature_transform_tag(),
		 				 options.numfeaturepartitions,
		 				 options.numthreads,
		 				 options.tolerance,
		 				 options.MAXITER,
		 				 options.rho);	 

	 		 else
	 			 Solver = new BlockADMMSolver(
	 					 context, 
	 					 loss	,
	 					 regularizer,	
	 					 options.lambda,
	 					 features,
	 					 skylark::ml::kernels::gaussian_t(X.Width(), options.kernelparam),
	 					 skylark::ml::fast_feature_transform_tag(),
	 					 options.numfeaturepartitions,
	 					 options.numthreads,
	 					 options.tolerance,
	 					 options.MAXITER,
	 					 options.rho);	 
	 	 	break;
	 		
	 	 default:
	 		// TODO!
	 		break; 
	 	 
	 }

	 elem::Matrix<double> Wbar(features, k);
	 elem::MakeZeros(Wbar);

	 Solver->train(X, Y, Wbar);
	 if (context.rank==0) {
		 std::stringstream dimensionstring;
		 dimensionstring << "# Dimensions " << features << " " << k << "\n";
		 elem::Write(Wbar, options.print().append(dimensionstring.str()), options.modelfile);
	 }
	 
	 // Testing - if specified by the user.
	 if (!options.testfile.empty()) {
		 context.comm.barrier();
		 
		 if(context.rank == 0) std::cout << "Starting testing phase." << std::endl;
		 
		 DistInputMatrixType Xt, Yt;
		 read_libsvm_dense(context, options.testfile, Xt, Yt, X.Width());
		 
		 DistTargetMatrixType Yp(Yt.Height(), k);
		 Solver->predict(Xt, Yp, Wbar);
		 		 
		 int correct = 0;
		 double o, o1;
		 int pred;
		 for(int i=0; i < Yp.LocalHeight(); i++) {
			 o = Yp.GetLocal(i,0);
			 pred = 0;
			 for(int j=1; j < Yp.Width(); j++) {
				 o1 = Yp.GetLocal(i,j);
				 if ( o1 > o) {
					 o = o1;
					 pred = j;
				 }
			 }

			 if(pred == (int) Yt.GetLocal(i,0))
				 correct++;
		  }	
		
		 int totalcorrect;
		 boost::mpi::reduce(context.comm, correct, totalcorrect, std::plus<double>(), 0);
		 if(context.rank ==0)
			 std::cout << "Accuracy = " << totalcorrect*100.0/Xt.Height() << " %" << std::endl;

	 } 
	 
	context.comm.barrier();
	 
	 elem::Finalize();
	 return 0;
}
