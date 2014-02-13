/*
 * FunctionProx.hpp
 *
 *  Created on: Jan 12, 2014
 *      Author: vikas
 */

#ifndef FUNCTIONPROX_HPP_
#define FUNCTIONPROX_HPP_

#include <elemental.hpp>
#include "options.hpp"
#include <cstdlib>
#include <omp.h>

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

// Class to represent hinge loss
class hingeloss : public lossfunction {
public:
	virtual double evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T);
	virtual void proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y);
};

class logisticloss : public lossfunction {
public:
    virtual double evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T);
    virtual void proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y);
private:
    double logsumexp(double* x, int n);
    double normsquare(double* x, double* y, int n);
    double objective(int index, double* x, double* v, int n, double lambda);
    int logexp(int index, double* v, int n, double lambda, double* x, int MAXITER, double epsilon, int DISPLAY);
    static const int MAXITER = 100;
    static const double epsilon = 1e-4;
    static const int DISPLAY = 0;
};


class l2: public regularization {
public:
	virtual double evaluate(LocalDenseMatrixType& W);
	virtual void proxoperator(LocalDenseMatrixType& W, double lambda, LocalDenseMatrixType& mu, LocalDenseMatrixType& P);
};

double squaredloss::evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T) {
		double loss = 0.0;
		int k = O.Height();
		int n = O.Width();

		// check for size compatability

		double* Obuf = O.Buffer();
		double*  Tbuf = T.Buffer();
		double x;
		int i, j, label;
	
		if (k==1) {
			#pragma omp parallel for reduction(+:loss) private(i, x)
			for(i=0; i<n; i++) {
				x = Obuf[i] - Tbuf[i];
				loss += x*x;
			}
		}
		
		if (k>1) {
			#pragma omp parallel for reduction(+:loss) private(i,j, x, label)
			for(i=0; i<n; i++) {
				label = (int) Tbuf[i];
                                for(j=0;j<k;j++) {
                                     x = O.Get(j,i) - (j==label ? 1.0:-1.0);
                        	     loss += x*x;
	                       	}			
			}
		}

		return 0.5*loss;
	}

	//solution to Y = prox[X] = argmin_Y 0.5*||X-Y||^2_{fro} + lambda 0.5 ||Y-T||^2_{fro}
void squaredloss::proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y) {
		int k = X.Height();
		int n = X.Width();

				// check for size compatability

		double* Xbuf = X.Buffer();
		double* Tbuf = T.Buffer();

		double* Ybuf = Y.Buffer();
		double ilambda = 1.0/(1.0 + lambda);

		int label, i, j;

		if (k==1) {
			for(int i=0; i<n; i++)
				Ybuf[i] = ilambda*(Xbuf[i] + lambda*Tbuf[i]);
		}

		if(k>1) {
			for(int i=0; i<n; i++) {
				label = (int) Tbuf[i];
                                for(j=0;j<k;j++) {
                                     Y.Set(j, i,  ilambda*(X.Get(j,i) + lambda*(j==label ? 1.0:-1.0)));			
				}
			}			
		}
	}


double hingeloss::evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T) {
		double loss = 0.0;
		int k = O.Height();
		int n = O.Width();
		int kn = O.Height()*O.Width();
		int label, i, j;
		// check for size compatability

		double* Obuf = O.Buffer();
		double* Tbuf = T.Buffer();
		double obj = 0.0;
		double yx;

		int noutputs = O.Width();

		if(noutputs==1) {
               #pragma omp parallel for reduction(+:obj) private(i, yx)
		       for(i=0;i<n;i++) {
		                        yx = Obuf[i]*Tbuf[i];
		                        if(yx<1.0)
		                                obj += (1.0 - yx);
		                }

		        }


		if(noutputs>1) {
               #pragma omp parallel for reduction(+:obj) private(i, j, label, yx)
		       for(i=0;i<n;i++) {
		                label = (int) Tbuf[i];
		                for(j=0;j<k;j++) {
		                     yx = O.Get(j,i)* (j==label ? 1.0:-1.0);
		                     if(yx<1.0)
		                         obj += (1.0 - yx);
		                }
		       }

		}
		return obj;
	}

	//solution to Y = prox[X] = argmin_Y 0.5*||X-Y||^2_{fro} + lambda 0.5 ||Y-T||^2_{fro}
void hingeloss::proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y) {

	int i, j;
	double* Tbuf = T.Buffer();
	double* Xbuf = X.Buffer();
	double* Ybuf = Y.Buffer();
	int k = X.Height();
	int m = X.Width();
    double yv, yy;
    int label;

    int noutputs = k;

	if(noutputs==1) { // We assume cy has +1 or -1 entries for n=1 outputs
                        #pragma omp parallel for private(i,yv)
		                for(i=0;i<m;i++) {
		                        yv = Tbuf[i]*Xbuf[i];

		                        if (yv > 1.0) {
		                                Ybuf[i] = Xbuf[i];
		                        }
		                        else {
		                                if(yv < (1.0-lambda)) {
		                                        Ybuf[i] = Xbuf[i] + lambda*Tbuf[i];
		                                }
		                                else {
		                                        Ybuf[i] = Tbuf[i];
		                                }
		                        }
		                }
		        }

	if (noutputs>1) {
                        #pragma omp parallel for private(i,j,yv, yy, label)
		                for(i=0;i<m;i++) {
		                        label = (int) Tbuf[i];
		                        for(j=0;j<k;j++) {
		                                yv = X.Get(j,i);
		                                yy = +1.0;
		                                if(!(j==label)) {
		                                        yv = -yv;
		                                        yy = -1.0;
		                                }
		                                if (yv>1.0)
		                                                                Y.Set(j,i,  X.Get(j,i));
		                                                        else {
		                                                                if(yv<1.0-lambda)
		                                                                        Y.Set(j,i, X.Get(j,i) + lambda*yy);
		                                                                else
		                                                                        Y.Set(j,i, yy);
		                                                        }
		                        }
		                }
		        }


	}


double logisticloss::evaluate(LocalDenseMatrixType& O, LocalTargetMatrixType& T) {
        double obj = 0.0;
        int m = O.Width();
        int n = O.Height();

        double start = omp_get_wtime( );

        #pragma omp parallel for reduction(+:obj)
        for(int i=0;i<m;i++) {
            obj += -O.Get((int) T.Get(i, 0), i) + logsumexp(O.Buffer(0, i), n);
        }

        double end = omp_get_wtime( );

        // std::cout << end - start <<  " secs " << std::endl;

        return obj;
}


void logisticloss::proxoperator(LocalDenseMatrixType& X, double lambda, LocalTargetMatrixType& T, LocalDenseMatrixType& Y) {
    int flag = 0;
    int m = X.Width();
    int n = X.Height();

    #pragma omp parallel for
    for(int i=0;i<m;i++) {
                flag = logexp((int) T.Get(i, 0), X.Buffer(0, i), n, lambda, Y.Buffer(0, i), MAXITER, epsilon, DISPLAY);
    }

}

double logisticloss::logsumexp(double* x, int n) {
        int i;
        double max=x[0];
        double f = 0.0;
        for(i=0;i<n;i++) {
                if (x[i]>max) {
                        max = x[i];
                }
        }
        for(i=0;i<n;i++)
                f +=  exp(x[i] - max);
        f = max + log(f);

        return f;
}

double logisticloss::objective(int index, double* x, double* v, int n, double lambda) {
        double nrmsqr = normsquare(x,v,n);
        double obj = -x[index] + logsumexp(x, n) + 0.5*lambda*nrmsqr;
        return obj;
        }

double logisticloss::normsquare(double* x, double* y, int n){
        double nrm = 0.0;
        int i;
        for(i=0;i<n;i++)
                nrm+= pow(x[i] - y[i], 2);
        return nrm;
}

int logisticloss::logexp(int index, double* v, int n, double lambda, double* x, int MAXITER, double epsilon, int DISPLAY) {
    /* solution to - log exp(x(i))/sum(exp(x(j))) + lambda/2 ||x - v||_2^2 */
    /* n is length of v and x */
    /* writes over x */
    double alpha = 0.1;
    double beta = 0.5;
    int iter, i;
    double t, logsum, p, pu, pptil, decrement;
    double *u = (double *) malloc(n*sizeof(double));
    double *z = (double *) malloc(n*sizeof(double));
    double *grad = (double *) malloc(n*sizeof(double));
    double newobj=0.0, obj=0.0;
    obj = objective(index, x, v, n, lambda);

    for(iter=0;iter<MAXITER;iter++) {
        logsum = logsumexp(x,n);
        if(DISPLAY)
            printf("iter=%d, obj=%f\n", iter, obj);
        pu = 0.0;
        pptil = 0.0;
        for(i=0;i<n;i++) {
            p = exp(x[i] - logsum);
            grad[i] = p + lambda*(x[i] - v[i]);
            if(i==index)
                grad[i] += -1.0;
            u[i] = grad[i]/(p+lambda);
            pu += p*u[i];
            z[i] = p/(p+lambda);
            pptil += z[i]*p;
        }
        decrement = 0.0;
        for(i=0;i<n;i++) {
            u[i] -= (pu/pptil)*z[i];
            decrement += grad[i]*u[i];
        }
        if (decrement < 2*epsilon) {
            free(u);
            free(z);
            free(grad);
            return 0;
        }
        t = 1.0;
        while(1) {
            for(i=0;i<n;i++)
                z[i] = x[i] - t*u[i];
            newobj = objective(index, z, v, n, lambda);
            if (newobj <= obj + alpha*t*decrement)
                break;
            t = beta*t;
        }
        for(i=0;i<n;i++)
            x[i] = z[i];
            obj = newobj;
    }
    free(u);
    free(z);
    free(grad);
    return 1;
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
