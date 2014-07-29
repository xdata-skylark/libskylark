#ifndef SKYLARK_ASYNCH_PRECOND_HPP
#define SKYLARK_ASYNCH_PRECOND_HPP

namespace skylark { namespace algorithms {

template<typename MatType, typename RhsType, typename SolType>
struct asy_precond_t :
    public outplace_precond_t<RhsType, SolType> {


    asy_precond_t(const MatType& A, asy_iter_params_t params
        , base::context_t &context)
        : _A(A), _params(params), _context(context) { }

    bool is_id() const { return false; }

    void apply(const RhsType& B, SolType& X) const {
        elem::MakeZeros(X);
        AsyRGS(_A, B, X, _context, _params);
    }

    void apply_adjoint(const RhsType& B, SolType& X) const {
        // TODO
    }

private:
    const MatType& _A;
    const asy_iter_params_t _params;
    base::context_t &_context;
};

} }
#endif
