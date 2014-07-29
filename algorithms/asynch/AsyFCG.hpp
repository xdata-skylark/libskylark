#ifndef SKYLARK_ASYFCG_HPP
#define SKYLARK_ASYFCG_HPP

namespace skylark {
namespace algorithms {

template<typename MatType, typename RhsType, typename SolType>
void AsyFCG(const MatType& A, const RhsType& B, SolType& X,
    base::context_t& context,
    asy_iter_params_t params = asy_iter_params_t()) {

    krylov_iter_params_t krylov_params;
    krylov_params.iter_lim = params.iter_lim;
    krylov_params.res_print = params.iter_res_print;
    krylov_params.tolerance = params.tolerance;
    krylov_params.am_i_printing = params.am_i_printing;
    krylov_params.log_level = params.log_level;

    asy_iter_params_t asy_params;
    asy_params.tolerance = 0;
    asy_params.sweeps_lim = params.sweeps_lim;
    asy_params.syn_sweeps = params.syn_sweeps;

    FlexibleCG(A, B, X, krylov_params,
        asy_precond_t<MatType, RhsType, SolType>(A, asy_params, context));
}

} } // namespace skylark::algorithms

#endif
