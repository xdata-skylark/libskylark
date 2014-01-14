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
typedef elem::Matrix<float> LocalInputMatrixType;

//Generalize this later to binary types
typedef elem::Matrix<float> LocalOutputMatrixType;

// need long-er type
typedef int Int;

class FeatureTransform
{
public:
	virtual LocalInputMatrixType& map(const LocalInputMatrixType& X, Int start, Int end) = 0 ;
	virtual ~FeatureTransform(void){}
};


#endif /* FEATURETRANSFORM_HPP_ */
