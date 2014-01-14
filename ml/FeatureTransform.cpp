/*
 * FeatureTransform.cpp
 *
 *  Created on: Jan 13, 2014
 *      Author: vikas
 */

#include "FeatureTransform.hpp"

/* Identity transform for linear models */

class Identity: public FeatureTransform {
public:
	LocalInputMatrixType& map(LocalInputMatrixType& X, Int start, Int end) {
		// create a view attached to a location
		LocalInputMatrixType Z;
		elem::View(Z, X, 0, start, X.Height(), end - start + 1);
		return Z;
	}
};

/* To implement: RR and Fastfood and Rounding */
