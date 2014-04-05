/*
 * predict.cpp
 *
 *  Created on: Feb 7, 2014
 *      Author: vikas
 */

#include "hilbert.hpp"
#include <boost/mpi.hpp>
#include "../base/context.hpp"


#define DEBUG std::cout << "error " << std::endl;

int main (int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

	std::string testfile = argv[1];
	std::string modelfile = argv[2];
	boost::mpi::environment env (argc, argv);

        // get communicator
        boost::mpi::communicator comm;
        int rank = comm.rank();

        skylark::base::context_t context (12345);
	elem::Initialize (argc, argv);
	MPI_Comm mpi_world(comm);
	DistInputMatrixType X, Y;
	int m,n;
	elem::Matrix<double> W;


	if (rank == 0) {

		read_model_file(modelfile, W);
		m = W.Height();
		n = W.Width();
	}
	boost::mpi::broadcast(comm, m, 0);
	boost::mpi::broadcast(comm, n, 0);
	if (rank != 0) {
		W.Resize(m,n);
	}

	boost::mpi::broadcast(comm, W.Buffer(), m*n, 0);

	read_libsvm_dense(context, testfile, X, Y);

	elem::DistMatrix<double, elem::VC, elem::STAR>  O(X.Height(), W.Width());
	elem::MakeZeros(O);

	elem::Gemm(elem::NORMAL,elem::NORMAL,1.0, X.Matrix(), W, 0.0, O.Matrix());

	int correct = 0;
	double o, o1;
	int pred;
	for(int i=0; i<O.LocalHeight(); i++) {
		o = O.GetLocal(i,0);
		pred = 0;
		for(int j=1; j<O.Width(); j++) {
			o1 = O.GetLocal(i,j);
			if ( o1 > o) {
				o = o1;
				pred = j;
			}
		}
		if(pred== (int) Y.GetLocal(i,0))
			correct++;
	}

	comm.barrier();

	//if(rank ==0) {
	int totalcorrect;
	boost::mpi::reduce(comm, correct, totalcorrect, std::plus<double>(), 0);
	if(rank ==0)
	    std::cout << "Accuracy = " << totalcorrect*100.0/X.Height() << " %" << std::endl;
	//}
}
