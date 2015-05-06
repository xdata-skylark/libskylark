#ifndef FUNCTIONPROX_HPP_
#define FUNCTIONPROX_HPP_

#include <El.hpp>

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif

struct loss_t {

    virtual
    double evaluate(El::Matrix<double>& O, El::Matrix<double>& T) const = 0;

    virtual void proxoperator(El::Matrix<double>& X, double lambda,
        El::Matrix<double>& T, El::Matrix<double>& Y) const = 0 ;

    virtual ~loss_t() {

    }
};

/**
 * Square loss: 0.5*||O - T||^2_{fro}
 */
struct squared_loss_t : public loss_t {

    virtual double evaluate(El::Matrix<double>& O, El::Matrix<double>& T) const {
        double loss = 0.0;
        int k = O.Height();
        int n = O.Width();

        // TODO: check for size compatability

        double* Obuf = O.Buffer();
        double* Tbuf = T.Buffer();

        if (k == 1) {
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:loss)
#           endif
            for(int i = 0; i < n; i++) {
                double x = Obuf[i] - Tbuf[i];
                loss += x*x;
            }
        } else {
            // k > 1: treat it as classification for now...
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:loss)
#           endif
            for(int i = 0; i < n; i++) {
                int label = (int) Tbuf[i];
                for(int j = 0;j < k; j++) {
                    double x = O.Get(j,i) - (j == label ? 1.0 : -1.0);
                    loss += x*x;
                }
            }
        }

        return 0.5*loss;
    }

    virtual void proxoperator(El::Matrix<double>& X, double lambda,
        El::Matrix<double>& T, El::Matrix<double>& Y) const {

        int k = X.Height();
        int n = X.Width();

        // TODO: check for size compatability

        double* Xbuf = X.Buffer();
        double* Tbuf = T.Buffer();

        double* Ybuf = Y.Buffer();
        double ilambda = 1.0 / (1.0 + lambda);

        if (k==1) {
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i = 0; i < n; i++)
                Ybuf[i] = ilambda * (Xbuf[i] + lambda * Tbuf[i]);
        } else {
            // k > 1: treat it as classification for now...
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i = 0; i < n; i++) {
                int label = (int) Tbuf[i];
                for(int j = 0; j < k; j++)
                    Y.Set(j, i,
                        ilambda*(X.Get(j,i) + lambda*(j==label ? 1.0:-1.0)));
            }
        }
    }
};

/**
 * Least absolute deviations loss: ||O - T||_1
 */
struct lad_loss_t : public loss_t {

    virtual double evaluate(El::Matrix<double>& O, El::Matrix<double>& T) const {
        double loss = 0.0;
        int k = O.Height();
        int n = O.Width();

        // TODO check for size compatability

        double* Obuf = O.Buffer();
        double* Tbuf = T.Buffer();

        if (k==1) {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:loss)
#           endif
            for(int i = 0; i < n; i++) {
                double x = Obuf[i] - Tbuf[i];
                loss += std::abs(x);
            }

        } else {
            // k > 1: treat it as classification for now...
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:loss)
#           endif
            for(int i = 0; i < n; i++) {
                int label = (int) Tbuf[i];
                for(int j = 0; j < k; j++) {
                    double x = O.Get(j,i) - (j==label ? 1.0:-1.0);
                    loss += std::abs(x);
                }
            }
        }
        return loss;
    }

    virtual void proxoperator(El::Matrix<double>& X, double lambda,
        El::Matrix<double>& T, El::Matrix<double>& Y) const {

        int k = X.Height();
        int n = X.Width();

        // TODO: check for size compatability

        double* Xbuf = X.Buffer();
        double* Tbuf = T.Buffer();

        double* Ybuf = Y.Buffer();
        double ilambda = 1.0/(1.0 + lambda);


        if (k==1) {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i=0; i<n; i++) {
                Ybuf[i] = Tbuf[i];
                if (Xbuf[i] > (Tbuf[i] + lambda))
                    Ybuf[i] = Xbuf[i] - lambda;
                if (Xbuf[i] < (Tbuf[i] - lambda))
                    Ybuf[i] = Xbuf[i] + lambda;
            }

        } else {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i = 0; i < n; i++) {
                int label = (int) Tbuf[i];
                for(int j = 0; j < k; j++) {
                    double t = (j==label ? 1.0:-1.0);
                    double x = X.Get(j,i);
                    Y.Set(j, i, t);
                    if (x > t + lambda)
                        Y.Set(j, i,  x - lambda);
                    if (x < t - lambda)
                        Y.Set(j, i,  x + lambda);
                }
            }

        }
    }
};

/**
 * Hinge-loss: sum(max(1 - t * o, 0))
 */
struct hinge_loss_t : public loss_t {

    virtual double evaluate(El::Matrix<double>& O, El::Matrix<double>& T) const {

        int k = O.Height();
        int n = O.Width();
        int kn = O.Height()*O.Width();

        // TODO: check for size compatability

        double* Obuf = O.Buffer();
        double* Tbuf = T.Buffer();
        double obj = 0.0;

        if (k==1) {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:obj)
#           endif
            for(int i = 0; i < n; i++) {
                double yx = Obuf[i]*Tbuf[i];
                if(yx<1.0)
                    obj += (1.0 - yx);
            }

        } else {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:obj)
#           endif
            for(int i = 0; i < n; i++) {
                int label = (int) Tbuf[i];
                for(int j = 0; j < k; j++) {
                    double yx = O.Get(j,i) * (j==label ? 1.0 : -1.0);
                    if(yx<1.0)
                        obj += (1.0 - yx);
                }
            }

        }

        return obj;
    }

    virtual void proxoperator(El::Matrix<double>& X, double lambda,
        El::Matrix<double>& T, El::Matrix<double>& Y) const {

        double* Tbuf = T.Buffer();
        double* Xbuf = X.Buffer();
        double* Ybuf = Y.Buffer();

        int k = X.Height();
        int n = X.Width();

        if (k==1) { // We assume cy has +1 or -1 entries for n=1 outputs

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i = 0; i < n; i++) {
                double yv = Tbuf[i] * Xbuf[i];
                if (yv > 1.0)
                    Ybuf[i] = Xbuf[i];
                else {
                    if(yv < (1.0-lambda))
                        Ybuf[i] = Xbuf[i] + lambda * Tbuf[i];
                    else
                        Ybuf[i] = Tbuf[i];

                }
            }

        } else {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i = 0; i < n; i++) {
                int label = (int) Tbuf[i];
                for(int j = 0; j < k; j++) {
                    double yv = X.Get(j,i);
                    double yy = +1.0;
                    if(!(j==label)) {
                        yv = -yv;
                        yy = -1.0;
                    }
                    if (yv > 1.0)
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
};

struct logistic_loss_t : public loss_t {

    virtual double evaluate(El::Matrix<double>& O, El::Matrix<double>& T) const {

        double loss = 0.0;
        int m = O.Width();
        int n = O.Height();

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for reduction(+:loss)
#       endif
        for(int i = 0; i < m; i++) {
            int t = (int) T.Get(i, 0);
            loss += -O.Get(t, i) + logsumexp(O.Buffer(0, i), n);
        }

        return loss;

    }

    virtual void proxoperator(El::Matrix<double>& X, double lambda, 
        El::Matrix<double>& T, El::Matrix<double>& Y) const {

        int m = X.Width();
        int n = X.Height();

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int i=0;i<m;i++) {
            int t = (int) T.Get(i, 0);
            logexp(t, X.Buffer(0, i), n, 1.0/lambda, Y.Buffer(0, i));
        }
    }

private:

    // TODO might be useful to make this an independent function
    static double logsumexp(double* x, int n) {
        double max = x[0];
        double f = 0.0;
        for(int i=0;i<n;i++)
            max = std::max(max, x[i]);

        for(int i = 0; i < n; i++)
            f += std::exp(x[i] - max);

        return max + log(f);
    }

    // Solution to - log exp(x(i))/sum(exp(x(j))) + lambda/2 ||x - v||_2^2 
    static int logexp(int index, double* v, int n, double lambda, double* x) {
        const int MAXITER = 100;
        const double epsilon = 1e-4;

        double alpha = 0.1;
        double beta = 0.5;
        double t, p, decrement;
        double *u = (double *) malloc(n*sizeof(double));
        double *z = (double *) malloc(n*sizeof(double));
        double *grad = (double *) malloc(n*sizeof(double));
        double newobj=0.0, obj=0.0;
        obj = objective(index, x, v, n, lambda);

        for(int iter = 0; iter < MAXITER; iter++) {
            double logsum = logsumexp(x,n);
            double pu = 0.0;
            double pptil = 0.0;
            for(int i = 0; i < n; i++) {
                p = exp(x[i] - logsum);
                grad[i] = p + lambda * (x[i] - v[i]);
                if (i == index)
                    grad[i] += -1.0;
                u[i] = grad[i]/(p+lambda);
                pu += p*u[i];
                z[i] = p/(p+lambda);
                pptil += z[i]*p;
            }

            pptil = 1 - pptil;
            double decrement = 0.0;
            for(int i=0; i < n;i++) {
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
                for(int i = 0; i < n; i++)
                    z[i] = x[i] - t*u[i];
                newobj = objective(index, z, v, n, lambda);
                if (newobj <= obj + alpha*t*decrement)
                    break;
                t = beta*t;
            }
            for(int i = 0; i < n; i++)
                x[i] = z[i];
            obj = newobj;
        }

        free(u);
        free(z);
        free(grad);
        return 1;
    }

    static double normsquare(double* x, double* y, int n) {
        double nrm = 0.0;
        for(int i = 0; i < n; i++)
            nrm += pow(x[i] - y[i], 2);
        return nrm;
    }


    static double objective(int index, double* x, 
        double* v, int n, double lambda) {

        double nrmsqr = normsquare(x,v,n);
        double obj = -x[index] + logsumexp(x, n) + 0.5*lambda*nrmsqr;
        return obj;
    }

};



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
