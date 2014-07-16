#ifndef SKYLARK_ASYFCG_HPP
#define SKYLARK_ASYFCG_HPP

namespace skylark {
namespace algorithms {

template<typename MatType, typename RhsType, typename SolType>
void AsyFCG(const MatType& A, const RhsType& B, SolType& X, int sweeps,
    base::context_t& context) {
    krylov_iter_params_t iter_params;
    iter_params.iter_lim = 100;
    FlexibleCG(A, B, X, iter_params,
        asy_precond_t<MatType, RhsType, SolType>(A, sweeps, context));
}

} } // namespace skylark::algorithms

#endif
