/*
 * FeatureTransform.cpp
 *
 *  Created on: Jan 13, 2014
 *      Author: vikas
 */

#include "FeatureTransform.hpp"

/* Identity transform for linear models */
void Identity::map(LocalInputMatrixType& X, Int start, Int end, LocalInputMatrixType& Z) {
		// create a view attached to a location
		elem::View(Z, X, 0, start, X.Height(), end - start + 1);
}

/* To implement: RR and Fastfood and Rounding */
