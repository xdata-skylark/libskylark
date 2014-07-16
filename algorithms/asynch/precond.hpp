#ifndef SKYLARK_ASYNCH_PRECOND_HPP
#define SKYLARK_ASYNCH_PRECOND_HPP

namespace skylark { namespace algorithms {

template<typename MatType, typename RhsType, typename SolType>
struct asy_precond_t :
    public outplace_precond_t<RhsType, SolType> {


    asy_precond_t(const MatType& A, int sweeps, base::context_t &context)
        : _A(A), _sweeps(sweeps), _context(context) { }

    bool is_id() const { return false; }

    void apply(const RhsType& B, SolType& X) const {
        elem::MakeZeros(X);
        AsyRGS(_A, B, X, _sweeps, _context);
    }

    void apply_adjoint(const RhsType& B, SolType& X) const {
        // TODO
    }

private:
    const MatType& _A;
    const int _sweeps;
    base::context_t &_context;
};

} }
#endif
