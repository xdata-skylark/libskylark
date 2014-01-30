/*
 * matrixutils.hpp
 *
 *  Created on: Jan 15, 2014
 *      Author: vikas
 */

#ifndef MATRIXUTILS_HPP_
#define MATRIXUTILS_HPP_

#include <elemental.hpp>


void matrixplus(double* a, double*b, int n) {
	for (int i=0; i<n; i++)
		a[i] += b[i];
}


void matrixminus(double* a, double*b, int n) {
	for (int i=0; i<n; i++)
		a[i] -= b[i];
}


#endif /* MATRIXUTILS_HPP_ */



