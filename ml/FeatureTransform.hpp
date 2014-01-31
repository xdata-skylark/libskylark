/*
 * FunctionProx.hpp
 *
 *  Created on: Jan 12, 2014
 *      Author: vikas
 */

#ifndef FEATURETRANSFORM_HPP_
#define FEATURETRANSFORM_HPP_

#include <elemental.hpp>

// Simple abstract class to represent a function and its prox operator
// these are defined for local matrices.
typedef elem::Matrix<double> LocalInputMatrixType;

//Generalize this later to binary types
typedef elem::Matrix<double> LocalOutputMatrixType;

// need long-er type
typedef int Int;

class FeatureTransform
{
public:
	virtual void map(LocalInputMatrixType& X, Int start, Int end, LocalInputMatrixType& Z) = 0 ;
	virtual ~FeatureTransform(void){}
};

class Identity: public FeatureTransform {
public:
	virtual void map(LocalInputMatrixType& X, Int start, Int end, LocalInputMatrixType& Z);
};

void Identity::map(LocalInputMatrixType& X, Int start, Int end, LocalInputMatrixType& Z) {
		// create a view attached to a location
		elem::View(Z, X, 0, start, X.Height(), end - start + 1);
}

#endif /* FEATURETRANSFORM_HPP_ */
