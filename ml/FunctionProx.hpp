/*
 * FunctionProx.hpp
 *
 *  Created on: Jan 12, 2014
 *      Author: vikas
 */

#ifndef FUNCTIONPROX_HPP_
#define FUNCTIONPROX_HPP_

#include <elemental.hpp>

// Simple abstract class to represent a function and its prox operator
// these are defined for local matrices.
typedef elem::Matrix<double> LocalDenseMatrixType;
typedef elem::Matrix<float>  LocalTargetMatrixType;

// abstract class for representing loss functions and their prox operators
class lossfunction
{
public:
	virtual double evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T) = 0 ;
	virtual void proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y) = 0 ;

	virtual ~lossfunction(void){}
};

// abstract class for representing regularizers and their prox operators
class regularization
{
public:
	virtual double evaluate(LocalDenseMatrixType& W) = 0 ;
	virtual void proxoperator(LocalDenseMatrixType& W, double lambda, LocalDenseMatrixType& P) = 0 ;

	virtual ~regularization(void){}
};



#endif /* FUNCTIONPROX_HPP_ */
