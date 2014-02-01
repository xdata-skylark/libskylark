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

typedef skylark::sketch::context_t skylark_context_t;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistInputMatrixType;
typedef elem::DistMatrix<double, elem::CIRC, elem::CIRC> DistCircMatrixType;

void read_libsvm_dense(skylark_context_t& context, string fName, DistInputMatrixType& X, DistInputMatrixType& Y) {
	if (context.rank==0)
			cout << "Reading from file " << fName << endl;

	ifstream file(fName.c_str());
	string line;
	string token, val, ind;
	float label;
	unsigned int start = 0;
	unsigned int delim, t;
	int n = 0;
	int d = 0;
	int i, j, last;
	char c;

	bmpi::timer timer;


	// make one pass over the data to figure out dimensions - will pay in terms of preallocated storage.
	while(!file.eof()) {
		getline(file, line);
		if(line.length()==0)
			break;
		delim = line.find_last_of(":");
		if(delim > line.length())
			continue;
		n++;
		t = delim;
		while(line[t]!=' ') {
			t--;
		}
		val = line.substr(t+1, delim - t);
		last = atoi(val.c_str());
		if (last>d)
			d = last;
	}

	DistCircMatrixType x(n, d), y(n, 1);
	x.SetRoot(0);
	y.SetRoot(0);

	if(context.rank==0) {
		double *Xdata = x.Matrix().Buffer();
		double *Ydata = y.Matrix().Buffer();

		// second pass
		file.clear();
		file.seekg(0, std::ios::beg);
		i = -1;
		while(!file.eof()) {
			getline(file, line);
			if( line.length()==0) {
				break;
			}
			i++;
			istringstream tokenstream (line);
			tokenstream >> label;
			Ydata[i] = label;

			while (tokenstream >> token)
			 {
				delim  = token.find(':');
				ind = token.substr(0, delim);
				val = token.substr(delim+1); //.substr(delim+1);
				j = atoi(ind.c_str()) - 1;
				Xdata[n*j + i] = atof(val.c_str());
			 }
		}
	}

	// The calls below should distribute the data to all the nodes.
	if (context.rank==0)
		cout << "Distributing Data.." << endl;

	X = x;
	Y = y;

	double readtime = timer.elapsed();
	if (context.rank==0)
		cout << "Read Matrix with dimensions: " << n << " by " << d << " (" << readtime << "secs)" << endl;
}

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

	 elem::Matrix<double> Wbar(options.randomfeatures, 1);
	 elem::MakeZeros(Wbar);
	 Solver->train(context, X,Y,Wbar);

	 elem::Write(Wbar, options.print(), options.modelfile);
//	 cout << " Rank " << context.rank << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << endl;

	 elem::Finalize();
	 return 0;
}
