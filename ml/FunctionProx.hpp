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
typedef elem::Matrix<double>  LocalTargetMatrixType;

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
	virtual void proxoperator(LocalDenseMatrixType& W, double lambda, LocalDenseMatrixType& mu, LocalDenseMatrixType& P) = 0 ;

	virtual ~regularization(void){}
};

// Class to represent 0.5*||O - T||^2_{fro}
class squaredloss : public lossfunction {
public:
	virtual double evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T);
	virtual void proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y);
};

class l2: public regularization {
public:
	virtual double evaluate(LocalDenseMatrixType& W);
	virtual void proxoperator(LocalDenseMatrixType& W, double lambda, LocalDenseMatrixType& mu, LocalDenseMatrixType& P);
};

double squaredloss::evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T) {
		double loss = 0.0;
		int mn = O.Height()*O.Width();

		// check for size compatability

		double* Obuf = O.Buffer();
		double*  Tbuf = T.Buffer();
		double x;

		for(int i=0; i<mn; i++) {
			x = Obuf[i] - Tbuf[i];
			loss += x*x;
		}

		return 0.5*loss;
	}

	//solution to Y = prox[X] = argmin_Y 0.5*||X-Y||^2_{fro} + lambda 0.5 ||Y-T||^2_{fro}
void squaredloss::proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y) {
		int mn = X.Height()*X.Width();

				// check for size compatability

		double* Xbuf = X.Buffer();
		double* Tbuf = T.Buffer();

		double* Ybuf = Y.Buffer();
		double ilambda = 1.0/(1.0 + lambda);

		for(int i=0; i<mn; i++)
			Ybuf[i] = ilambda*(Xbuf[i] + lambda*Tbuf[i]);
	}


double l2::evaluate(LocalDenseMatrixType& W) {
		double norm = elem::Norm(W);
		return 0.5*norm*norm;
	}


void l2::proxoperator(LocalDenseMatrixType& W, double lambda, LocalDenseMatrixType& mu, LocalDenseMatrixType& P) {
		double *Wbuf = W.Buffer();
		double *mubuf = mu.Buffer();
		double *Pbuf = P.Buffer();
		int mn = W.Height()*W.Width();
		double ilambda = 1.0/(1.0 + lambda);

		for(int i=0;i<mn; i++)
			Pbuf[i] = (Wbuf[i] - mubuf[i])*ilambda;
	}


#endif /* FUNCTIONPROX_HPP_ */
