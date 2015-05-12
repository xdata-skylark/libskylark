#ifndef SKYLARK_REGULARIZERS_HPP
#define SKYLARK_REGULARIZERS_HPP

namespace skylark { namespace algorithms {

template<typename ValueType>
struct regularizer_t
{
    typedef ValueType value_type;

    virtual double evaluate(const El::Matrix<ValueType>& W) const = 0;

    virtual void proxoperator(const El::Matrix<ValueType>& W, double lambda, 
        const El::Matrix<ValueType>& mu, El::Matrix<ValueType>& P) const = 0 ;

    virtual ~regularizer_t() {

    }
};

template<typename ValueType>
struct empty_regularizer_t : public regularizer_t<ValueType> {

    virtual double evaluate(const El::Matrix<ValueType>& W) const {

        return 0.0;
    }

    virtual void proxoperator(const El::Matrix<ValueType>& W, double lambda,
        const El::Matrix<ValueType>& mu, El::Matrix<ValueType>& P) const {

        El::Copy(W, P);
        El::Axpy(-1.0, mu, P);
    }
};

template<typename ValueType>
struct l2_regularizer_t : public regularizer_t<ValueType> {

    virtual double evaluate(const El::Matrix<ValueType>& W) const {

        double norm = El::Norm(W);
        return 0.5*norm*norm;
    }

    virtual void proxoperator(const El::Matrix<ValueType>& W, double lambda,
        const El::Matrix<ValueType>& mu, El::Matrix<ValueType>& P) const {

        const ValueType *Wbuf = W.LockedBuffer();
        const ValueType *mubuf = mu.LockedBuffer();
        ValueType *Pbuf = P.Buffer();
        El::Int mn = W.Height() * W.Width();
        double ilambda = 1.0/(1.0 + lambda);

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(El::Int i=0; i < mn; i++)
            Pbuf[i] = (Wbuf[i] - mubuf[i]) * ilambda;
    }
};

template<typename ValueType>
struct l1_regularizer_t : public regularizer_t<ValueType> {

    virtual double evaluate(const El::Matrix<ValueType>& W) const {

        double norm = El::EntrywiseNorm(W, 1);
        return norm;
    }

    virtual void proxoperator(const El::Matrix<ValueType>& W, double lambda,
        const El::Matrix<ValueType>& mu, El::Matrix<ValueType>& P) const {

        const ValueType *Wbuf = W.LockedBuffer();
        const ValueType *mubuf = mu.LockedBuffer();
        ValueType *Pbuf = P.Buffer();
        El::Int mn = W.Height() * W.Width();

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for
#       endif
        for(El::Int i = 0; i < mn; i++)
            Pbuf[i] = soft_threshold(Wbuf[i] - mubuf[i], lambda);

    }

private:
    ValueType soft_threshold(ValueType x, ValueType lambda) const {
        ValueType v = 0;
        if (std::abs(x) <= lambda)
            v = 0.0;
        if (x > lambda)
            v =  x - lambda;
        if (x < -lambda)
            v = x + lambda;
        return v;
    }
};

} } // namespace skylark::algorithms

#endif /* SKYLARK_REGULARIZERS_HPP */
