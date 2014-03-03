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

	  bmpi::environment env (argc, argv);

	/* Create a global communicator */
	  bmpi::communicator world;


	 skylark::sketch::context_t context (12345, world);



	 elem::Initialize (argc, argv);

	 hilbert_options_t options (argc, argv, context.size);
	 if (options.exit_on_return) { return -1; }

	 MPI_Comm mpi_world(world);

	 //elem::Grid grid (mpi_world);

#ifdef SKYLARK_HAVE_OPENMP
	 omp_set_num_threads(options.numthreads);
#endif

	 DistInputMatrixType X;
	 DistTargetMatrixType Y;

	 if (context.rank==0)
		 std::cout << options.print();

	 switch(options.fileformat) {
	     case LIBSVM:
	         read_libsvm_dense(context, options.trainfile, X, Y);
	         break;

	     case HDF5:
#ifdef SKYLARK_HAVE_HDF5
	         read_hdf5_dense(context, options.trainfile, X, Y);
#else
	         // TODO
#endif
	         break;
	 }
	 std::cout << " Rank " << context.rank << " on " << env.processor_name() << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << std::endl;

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
	 	 case LAD:
	 	 default:
	 		 // TODO
	 		 break;
	 }

	 regularization *regularizer = NULL;
	 switch(options.regularizer) {
	 	 case L2:
	 		 regularizer = new l2();
	 		 break;
	 	 case L1:
	 	 default:
	 		 // TODO
	 		 break;
	 }

	 // int k = Y.Width();
	 int k;
	 int kmax = *std::max_element(Y.Buffer(), Y.Buffer() + Y.LocalHeight());

	 boost::mpi::all_reduce(context.comm, kmax, k, boost::mpi::maximum<int>());
	 if (k>1) // we assume 0-to-N encoding of classes. Hence N = k+1. For two classes, k=1.
	 	k++;

	 BlockADMMSolver *Solver = NULL;
	 int features = 0;
	 switch(options.kernel) {
	 	 case LINEAR:
	 		 features = X.Height();
	 		 Solver = new BlockADMMSolver(
	 				context,
	 				loss,
	 				 regularizer,
	 				 options.lambda,
	 				 X.Height(),
	 				 options.numfeaturepartitions);
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
		 				 skylark::ml::kernels::gaussian_t(X.Height(), options.kernelparam),
		 				 skylark::ml::regular_feature_transform_tag(),
		 				 options.numfeaturepartitions);

	 		 else
	 			 Solver = new BlockADMMSolver(
	 					 context,
	 					 loss	,
	 					 regularizer,
	 					 options.lambda,
	 					 features,
	 					 skylark::ml::kernels::gaussian_t(X.Height(), options.kernelparam),
	 					 skylark::ml::fast_feature_transform_tag(),
	 					 options.numfeaturepartitions);
	 	 	break;

	 	 case POLYNOMIAL:
	 		 features = options.randomfeatures;
	 		 Solver = new BlockADMMSolver(
	 				 context,
	 				 loss,
	 				 regularizer,
	 				 options.lambda,
	 				 features,
	 				 skylark::ml::kernels::polynomial_t(X.Height(), options.kernelparam, options.kernelparam2, options.kernelparam3),
		 			 skylark::ml::regular_feature_transform_tag(),
		 			 options.numfeaturepartitions);
	 		 break;

	 	 case LAPLACIAN:
	 		 features = options.randomfeatures;
	 		 Solver = new BlockADMMSolver(
	 				 context,
	 				 loss,
	 				 regularizer,
	 				 options.lambda,
	 				 features,
	 				 skylark::ml::kernels::laplacian_t(X.Height(), options.kernelparam),
		 			 skylark::ml::regular_feature_transform_tag(),
		 			 options.numfeaturepartitions);
	 		 break;

	 	 case EXPSEMIGROUP:
	 		 features = options.randomfeatures;
	 		 Solver = new BlockADMMSolver(
	 				 context,
	 				 loss,
	 				 regularizer,
	 				 options.lambda,
	 				 features,
	 				 skylark::ml::kernels::expsemigroup_t(X.Height(), options.kernelparam),
		 			 skylark::ml::regular_feature_transform_tag(),
		 			 options.numfeaturepartitions);
	 		 break;

	 	 default:
	 		// TODO!
	 		break;

	 }

	 // Set parameters
	 Solver->set_rho(options.rho);
	 Solver->set_maxiter(options.MAXITER);
	 Solver->set_tol(options.tolerance);
	 Solver->set_nthreads(options.numthreads);

	 elem::Matrix<double> Wbar(features, k);
	 elem::MakeZeros(Wbar);

	 DistInputMatrixType Xv;
	 DistTargetMatrixType Yv;

	 if (!options.valfile.empty()) {
	          context.comm.barrier();

	          if(context.rank == 0) std::cout << "Loading validation data." << std::endl;

	          switch(options.fileformat) {
	                   case LIBSVM:
	                       read_libsvm_dense(context, options.valfile, Xv, Yv, X.Height());
	                       break;
#ifdef SKYLARK_HAVE_HDF5
	                   case HDF5:
	                       read_hdf5_dense(context, options.valfile, Xv, Yv);
	                       break;
#endif
	               }

	 }

	 Solver->train(X, Y, Wbar, Xv, Yv);


	 if (context.rank==0) {
		 std::stringstream dimensionstring;
		 dimensionstring << "# Dimensions " << features << " " << k << "\n";
		 elem::Write(Wbar, options.modelfile, elem::ASCII, options.print().append(dimensionstring.str()));
	 }

	 // Testing - if specified by the user.
	 if (!options.testfile.empty()) {
		 context.comm.barrier();

		 if(context.rank == 0) std::cout << "Starting testing phase." << std::endl;

		 DistInputMatrixType Xt;
		 DistTargetMatrixType Yt;

		 switch(options.fileformat) {
		                        case LIBSVM:
		                            read_libsvm_dense(context, options.testfile, Xt, Yt, X.Height());
		                            break;
#ifdef SKYLARK_HAVE_HDF5
		                        case HDF5:
		                            read_hdf5_dense(context, options.testfile, Xt, Yt);
		                            break;
#endif
		                    }

		 DistTargetMatrixType Yp(Yt.Height(), k);
		 Solver->predict(Xt, Yp, Wbar);
		 double accuracy = Solver->evaluate(Yt, Yp);

		 if(context.rank == 0) std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;
	 }

	context.comm.barrier();

	 elem::Finalize();
	 return 0;
}
