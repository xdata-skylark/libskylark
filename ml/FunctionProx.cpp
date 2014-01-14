/*
 * FunctionProx.cpp
 *
 *  Created on: Jan 12, 2014
 *      Author: Vikas Sindhwani (vsindhw@us.ibm.com)
 */

#include <elemental.hpp>
#include "FunctionProx.hpp"

// Class to represent 0.5*||O - T||^2_{fro}
class squaredloss : public lossfunction {
public:
	double evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T) {
		double loss = 0.0;
		int mn = O.Height()*O.Width();

		// check for size compatability

		double* Obuf = O.Buffer();
		float*  Tbuf = T.Buffer();
		double x;

		for(int i=0; i<mn; i++) {
			x = Obuf[i] - Tbuf[i];
			loss += x*x;
		}

		return 0.5*loss;
	}

	//solution to Y = prox[X] = argmin_Y 0.5*||X-Y||^2_{fro} + lambda 0.5 ||Y-T||^2_{fro}
	void proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y) {
		int mn = X.Height()*X.Width();

				// check for size compatability

		double* Xbuf = X.Buffer();
		float* Tbuf = T.Buffer();

		double* Ybuf = Y.Buffer();
		double ilambda = 1.0/(1.0 + lambda);

		for(int i=0; i<mn; i++)
			Ybuf[i] = ilambda*(Xbuf[i] + lambda*Ybuf[i]);
	}
};

class l2: public regularization {
public:
	double evaluate(LocalDenseMatrixType& W) {
		return 0.5*elem::Norm(W);
	}
	void proxoperator(LocalDenseMatrixType& W, double lambda, LocalDenseMatrixType& P) {
		double *Wbuf = W.Buffer();
		double *Pbuf = P.Buffer();
		int mn = W.Height()*W.Width();
		double ilambda = 1.0/(1.0 + lambda);

		for(int i=0;i<mn; i++)
			Pbuf[i] = Wbuf[i]*ilambda;
	}
};
