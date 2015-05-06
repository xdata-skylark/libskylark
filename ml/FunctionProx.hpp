#ifndef FUNCTIONPROX_HPP_
#define FUNCTIONPROX_HPP_

#include <El.hpp>

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif

// abstract class for representing regularizers and their prox operators
class regularization
{
public:
	virtual double evaluate(El::Matrix<double>& W) = 0 ;
	virtual void proxoperator(El::Matrix<double>& W, double lambda, El::Matrix<double>& mu, El::Matrix<double>& P) = 0 ;

	virtual ~regularization(void){}
};


class l2: public regularization {
public:
	virtual double evaluate(El::Matrix<double>& W);
	virtual void proxoperator(El::Matrix<double>& W, double lambda, El::Matrix<double>& mu, El::Matrix<double>& P);
};

class l1: public regularization {
public:
	virtual double evaluate(El::Matrix<double>& W);
	virtual void proxoperator(El::Matrix<double>& W, double lambda, El::Matrix<double>& mu, El::Matrix<double>& P);
	double soft_threshold(double x, double lambda);
};


double l2::evaluate(El::Matrix<double>& W) {
		double norm = El::Norm(W);
		return 0.5*norm*norm;
	}


void l2::proxoperator(El::Matrix<double>& W, double lambda, El::Matrix<double>& mu, El::Matrix<double>& P) {
		double *Wbuf = W.Buffer();
		double *mubuf = mu.Buffer();
		double *Pbuf = P.Buffer();
		int mn = W.Height()*W.Width();
		double ilambda = 1.0/(1.0 + lambda);

		for(int i=0;i<mn; i++)
			Pbuf[i] = (Wbuf[i] - mubuf[i])*ilambda;
	}

double l1::evaluate(El::Matrix<double>& W) {
    double norm = El::EntrywiseNorm(W, 1);
    return norm;
}

double l1::soft_threshold(double x, double lambda) {
	double v = 0;
	if (std::abs(x) <= lambda)
		v = 0.0;
	if (x > lambda)
		v =  x - lambda;
	if (x < -lambda)
		v = x + lambda;
	return v;
}

void l1::proxoperator(El::Matrix<double>& W, double lambda, El::Matrix<double>& mu, El::Matrix<double>& P) {
		double *Wbuf = W.Buffer();
		double *mubuf = mu.Buffer();
		double *Pbuf = P.Buffer();
		int mn = W.Height()*W.Width();

		for(int i=0;i<mn; i++)
			Pbuf[i] = soft_threshold(Wbuf[i] - mubuf[i], lambda);
	}


#endif /* FUNCTIONPROX_HPP_ */
