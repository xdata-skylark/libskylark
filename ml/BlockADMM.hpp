/*
 * BlockADMM.h
 *
 *  Created on: Jan 12, 2014
 *      Author: vikas
 */

#ifndef BLOCKADMM_H_
#define BLOCKADMM_H_

#include <elemental>
#include "FunctionProx.hpp"
#include "../skylark.hpp"

typedef elem::DistMatrix<float, elem::distribution_wrapper::VC, elem::distribution_wrapper::STAR> DistInputMatrixType;
typedef elem::DistMatrix<float, elem::distribution_wrapper::VC, elem::distribution_wrapper::STAR> DistTargetMatrixType;
typedef elem::Matrix<double> ModelType;
typedef skylark::sketch::context_t skylark_context_t;

class BlockADMMSolver
{
public:
	BlockADMMSolver(const lossfunction* loss,
					const regularization* regularizer,
					const FeatureTransform& featureMap,
					int NumFeaturePartitions = 1,
					int NumThreads = 1,
					double TOL = 0.1,
					int MAXITER = 1000,
					double RHO = 1.0);
	~BlockADMMSolver();

	int train(skylark_context_t& context, DistInputMatrixType& X, DistTargetMatrixType& Y, ModelType& W);

private:
	double RHO;
	int MAXITER;
	double TOL;
	int NumFeaturePartitions;
	int NumThreads;
	lossfunction* loss;
	regularization* regularizer;
	FeatureTransform* featureMap;
};



#endif /* BLOCKADMM_H_ */
