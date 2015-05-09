#ifndef SKYLARK_LOSS_HPP
#define SKYLARK_LOSS_HPP

namespace skylark { namespace algorithms {

template<typename ValueType>
struct loss_t {

    typedef ValueType value_type;

    virtual double evaluate(const El::Matrix<ValueType>& O,
        const El::Matrix<ValueType>& T) const = 0;

    virtual void proxoperator(const El::Matrix<ValueType>& X, double lambda,
        const El::Matrix<ValueType>& T, El::Matrix<ValueType>& Y) const = 0 ;

    virtual ~loss_t() {

    }
};

/**
 * Square loss: 0.5*||O - T||^2_{fro}
 */
template<typename ValueType>
struct squared_loss_t : public loss_t<ValueType> {

    typedef ValueType value_type;

    virtual
    double evaluate(const El::Matrix<ValueType>& O, 
        const  El::Matrix<ValueType>& T) const {

        double loss = 0.0;
        int k = O.Height();
        int n = O.Width();

        // TODO: check for size compatability

        const ValueType* Obuf = O.LockedBuffer();
        const ValueType* Tbuf = T.LockedBuffer();

        if (k == 1) {
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:loss)
#           endif
            for(int i = 0; i < n; i++) {
                ValueType x = Obuf[i] - Tbuf[i];
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
                    ValueType x = O.Get(j,i) - (j == label ? 1.0 : -1.0);
                    loss += x*x;
                }
            }
        }

        return 0.5*loss;
    }

    virtual void proxoperator(const El::Matrix<ValueType>& X, double lambda,
        const El::Matrix<ValueType>& T, El::Matrix<ValueType>& Y) const {

        int k = X.Height();
        int n = X.Width();

        // TODO: check for size compatability

        const ValueType* Xbuf = X.LockedBuffer();
        const ValueType* Tbuf = T.LockedBuffer();

        ValueType* Ybuf = Y.Buffer();
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
template<typename ValueType>
struct lad_loss_t : public loss_t<ValueType> {

    typedef ValueType value_type;

    virtual double evaluate(const El::Matrix<ValueType>& O, 
        const El::Matrix<ValueType>& T) const {

        double loss = 0.0;
        int k = O.Height();
        int n = O.Width();

        // TODO check for size compatability

        const ValueType* Obuf = O.LockedBuffer();
        const ValueType* Tbuf = T.LockedBuffer();

        if (k==1) {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:loss)
#           endif
            for(int i = 0; i < n; i++) {
                ValueType x = Obuf[i] - Tbuf[i];
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
                    ValueType x = O.Get(j,i) - (j==label ? 1.0:-1.0);
                    loss += std::abs(x);
                }
            }
        }
        return loss;
    }

    virtual void proxoperator(const El::Matrix<ValueType>& X, double lambda,
        const El::Matrix<ValueType>& T, El::Matrix<ValueType>& Y) const {

        int k = X.Height();
        int n = X.Width();

        // TODO: check for size compatability

        const ValueType* Xbuf = X.LockedBuffer();
        const ValueType* Tbuf = T.LockedBuffer();

        ValueType* Ybuf = Y.Buffer();
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
                    ValueType t = (j==label ? 1.0:-1.0);
                    ValueType x = X.Get(j,i);
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
template<typename ValueType>
struct hinge_loss_t : public loss_t<ValueType> {

    typedef ValueType value_type;

    virtual double evaluate(const El::Matrix<ValueType>& O,
        const El::Matrix<ValueType>& T) const {

        int k = O.Height();
        int n = O.Width();
        int kn = O.Height()*O.Width();

        // TODO: check for size compatability

        const ValueType* Obuf = O.LockedBuffer();
        const ValueType* Tbuf = T.LockedBuffer();
        double obj = 0.0;

        if (k==1) {

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for reduction(+:obj)
#           endif
            for(int i = 0; i < n; i++) {
                ValueType yx = Obuf[i]*Tbuf[i];
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
                    ValueType yx = O.Get(j,i) * (j==label ? 1.0 : -1.0);
                    if(yx<1.0)
                        obj += (1.0 - yx);
                }
            }

        }

        return obj;
    }

    virtual void proxoperator(const El::Matrix<ValueType>& X, double lambda,
        const El::Matrix<ValueType>& T, El::Matrix<ValueType>& Y) const {

        const ValueType* Tbuf = T.LockedBuffer();
        const ValueType* Xbuf = X.LockedBuffer();

        ValueType* Ybuf = Y.Buffer();

        int k = X.Height();
        int n = X.Width();

        if (k==1) { // We assume cy has +1 or -1 entries for n=1 outputs

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for
#           endif
            for(int i = 0; i < n; i++) {
                ValueType yv = Tbuf[i] * Xbuf[i];
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
                    ValueType yv = X.Get(j,i);
                    ValueType yy = +1.0;
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

template<typename ValueType>
struct logistic_loss_t : public loss_t<ValueType> {

    typedef ValueType value_type;

    virtual double evaluate(const El::Matrix<ValueType>& O,
        const El::Matrix<ValueType>& T) const {

        double loss = 0.0;
        int m = O.Width();
        int n = O.Height();

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for reduction(+:loss)
#       endif
        for(int i = 0; i < m; i++) {
            int t = (int) T.Get(i, 0);
            loss += -O.Get(t, i) + logsumexp(O.LockedBuffer(0, i), n);
        }

        return loss;

    }

    virtual void proxoperator(const El::Matrix<ValueType>& X, double lambda,
        const El::Matrix<ValueType>& T, El::Matrix<ValueType>& Y) const {

        int m = X.Width();
        int n = X.Height();

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(int i=0;i<m;i++) {
            int t = (int) T.Get(i, 0);
            logexp(t, X.LockedBuffer(0, i), n, 1.0/lambda, Y.Buffer(0, i));
        }
    }

private:

    // TODO might be useful to make this an independent function
    static double logsumexp(const ValueType* x, int n) {
        ValueType max = x[0];
        ValueType f = 0.0;
        for(int i=0;i<n;i++)
            max = std::max(max, x[i]);

        for(int i = 0; i < n; i++)
            f += std::exp(x[i] - max);

        return max + log(f);
    }

    // Solution to - log exp(x(i))/sum(exp(x(j))) + lambda/2 ||x - v||_2^2 
    static
    int logexp(int index, const ValueType* v, int n, double lambda, 
        ValueType* x) {

        const int MAXITER = 100;
        const ValueType epsilon = 1e-4;

        ValueType alpha = 0.1;
        ValueType beta = 0.5;
        ValueType l,t, p, decrement;
        ValueType *u = (double *) malloc(n*sizeof(double));
        ValueType *z = (double *) malloc(n*sizeof(double));
        ValueType *grad = (double *) malloc(n*sizeof(double));
        ValueType newobj=0.0, obj=0.0;
        obj = objective(index, x, v, n, lambda);

        for(int iter = 0; iter < MAXITER; iter++) {
            ValueType logsum = logsumexp(x,n);
            ValueType pu = 0.0;
            ValueType pptil = 0.0;
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
            ValueType decrement = 0.0;
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

    static ValueType normsquare(const double* x, const double* y, int n) {
        ValueType nrm = 0.0;
        for(int i = 0; i < n; i++)
            nrm += pow(x[i] - y[i], 2);
        return nrm;
    }


    static ValueType objective(int index, const double* x, 
        const ValueType* v, int n, double lambda) {

        ValueType nrmsqr = normsquare(x,v,n);
        ValueType obj = -x[index] + logsumexp(x, n) + 0.5*lambda*nrmsqr;
        return obj;
    }

};

} } // namespace skylark::algorithms

#endif // SKYLARK_LOSS_HPP
