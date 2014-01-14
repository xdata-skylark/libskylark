/*
 * hilbert.cpp
 *
 *  Created on: Jan 12, 2014
 *      Author: vikas
 */

#include <elemental.hpp>
#include "../sketch/context.hpp"
#include "BlockADMM.hpp"

namespace mpi boost::mpi;

typedef elem::DistMatrix<double, elem::distribution_wrapper::VC, elem::distribution_wrapper::STAR> DistMatrixType;
typedef elem::Matrix<double> LocalMatrixType;

BlockADMMSolver::BlockADMMSolver(const lossfunction* loss,
		const regularization* regularizer,
		const FeatureTransform& featureMap,
		int NumFeaturePartitions,
		int NumThreads,
		double TOL,
		int MAXITER,
		double RHO) {

		this->loss = loss;
		this->regularizer = regularizer;
		this->featureMap = featureMap;
		this->NumFeaturePartitions = NumFeaturePartitions;
		this->NumThreads = NumThreads;
		this->TOL = TOL;
		this->MAXITER = MAXITER;
		this->RHO = RHO;
}


int BlockADMMSolver::train(skylark_context_t& context,  DistInputMatrixType& X, DistTargetMatrixType& Y, LocalMatrixType& Wbar) {

	int P = context.size;

	int n = X.Height();
	int d = X.Width();

	// number of classes, targets - to generalize
	int k = Y.Width();

	// number of random features
	int D = Wbar.Height();

	elem::Grid grid = elem::DefaultGrid();

	DistMatrixType O(grid, n, k); //uses default Grid
	elem::Zeros(O, n, k);
	DistMatrixType Obar(grid, n, k); //uses default Grid
	elem::Zeros(Obar, n, k);
	DistMatrixType nu(grid, n, k); //uses default Grid
	elem::Zeros(nu, n, k);

	LocalMatrixType W, mu, Wi, mu_ij, ZtObar_ij;

	if(context.rank==0) {
		elem::Zeros(W, D, k);
		elem::Zeros(mu, D, k);
		}
	elem::Zeros(Wi, D, k);
	elem::Zeros(mu_ij, D, k);
	elem::Zeros(ZtObar_ij, D, k);

	int iter = 0;

	int ni = O.LocalHeight();

	elem::Matrix<float> x = X.Matrix();
	elem::Matrix<float> y = Y.Matrix();

	double localloss = loss->evaluate(O.Matrix(), y);
	double totalloss;

	while(iter<MAXITER) {
		iter++;

		reduce(context.comm, localloss, totalloss, std::plus<double>(), 0);
		if(context.rank==0) {

		}


	}


	return 0;
}

