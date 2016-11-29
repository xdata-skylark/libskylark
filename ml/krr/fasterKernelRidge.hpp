#ifndef SKYLARK_FASTER_KERNEL_RIDGE_HPP
#define SKYLARK_FASTER_KERNEL_RIDGE_HPP

#ifndef SKYLARK_KRR_HPP
#error "Include top-level krr.hpp instead of including individuals headers"
#endif

#include "KrrParams.hpp"
#include "FeaturedMapPrecond.hpp"
#include "../kernels/kernels.hpp"

namespace skylark { namespace ml {

// TODO(Jordi): Should we mix the logic with logging?
template<typename T, typename KernelType>
void FasterKernelRidge(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda,
    El::DistMatrix<T> &A, El::Int s, base::context_t &context,
    krr_params_t params = krr_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    boost::mpi::timer timer;

    // Compute kernel matrix
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Computing kernel matrix... ";
        params.log_stream.flush();
        timer.restart();
    }

    El::DistMatrix<T> K, D;

    // Hack for experiments!
    if (params.iter_lim == -1)
        goto skip_kernel_creation;

    SymmetricGram(El::LOWER, direction, k, X, K);

    // Add regularizer
    El::Ones(D, X.Width(), 1);
    El::UpdateDiagonal(K, lambda, D);

 skip_kernel_creation:

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Creating precoditioner... ";
        if (log_lev2)
            params.log_stream << std::endl;
        params.log_stream.flush();
        timer.restart();
    }

    algorithms::outplace_precond_t<El::DistMatrix<T> > *P;
    if (s == 0)
        P = new algorithms::outplace_id_precond_t<El::DistMatrix<T> >();
    else
        P = new feature_map_precond_t<El::DistMatrix<T> >(k, lambda,
            X, s, context, params);

    if (log_lev1 && !log_lev2)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    if (log_lev2)
        params.log_stream << params.prefix
                          << "Took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Solving linear equation... "
                          << std::endl;
        params.log_stream.flush();
        timer.restart();
    }


    // Solve
    if (params.iter_lim != -1) {
        algorithms::krylov_iter_params_t cg_params(params.tolerance, params.iter_lim,
            params.am_i_printing, params.log_level - 1, params.res_print, 
            params.log_stream, params.prefix + "\t");

        El::Zeros(A, X.Width(), Y.Width());
        algorithms::CG(El::LOWER, K, Y, A, cg_params, *P);
    } else {
        // Hack for experiments!
        El::Zeros(A, X.Width(), Y.Width());
        P->apply(Y, A);
    }

    if (log_lev1)
        params.log_stream  << params.prefix
                           <<"Took " << boost::format("%.2e") % timer.elapsed()
                           << " sec\n";

    delete P;

}

void FASTERKERNELRIDGE(base::direction_t direction, const kernel_t &k, 
    const boost::any &X, const boost::any &Y, double lambda,
    const boost::any &A, El::Int s, base::context_t &context,
    krr_params_t params = krr_params_t()) {                       

#define SKYLARK_FASTERKERNELRIDGE_ANY_APPLY_DISPATCH(XT, YT, AT)	\
    if (A.type() == typeid(AT*)) {                                      \
        if (X.type() == typeid(XT*)) {                                  \
            if (Y.type() == typeid(YT*)) {                              \
	      FasterKernelRidge(					\
                    direction,                                          \
                    k,                                                  \
                    *boost::any_cast<XT*>(X),                           \
                    *boost::any_cast<YT*>(Y),                           \
                    lambda,                                             \
                    *boost::any_cast<AT*>(A),				\
		    s,							\
		    context);						\
                return;                                                 \
            }                                                           \
            if (Y.type() == typeid(const YT*)) {                        \
	      FasterKernelRidge(					\
                    direction,                                          \
                    k,                                                  \
                    *boost::any_cast<XT*>(X),                           \
                    *boost::any_cast<YT*>(Y),                           \
                    lambda,                                             \
                    *boost::any_cast<AT*>(A),				\
		    s,							\
		    context);						\
                return;                                                 \
            }                                                           \
        }                                                               \
        if (X.type() == typeid(const XT*)) {                            \
            if (Y.type() == typeid(YT*)) {                              \
	      FasterKernelRidge(					\
                    direction,                                          \
                    k,                                                  \
                    *boost::any_cast<XT*>(X),                           \
                    *boost::any_cast<YT*>(Y),                           \
                    lambda,                                             \
                    *boost::any_cast<AT*>(A),				\
		    s,							\
		    context);						\
                return;                                                 \
            }                                                           \
                                                                        \
            if (Y.type() == typeid(const YT*)) {                        \
	      FasterKernelRidge(					\
                    direction,                                          \
                    k,                                                  \
                    *boost::any_cast<XT*>(X),                           \
                    *boost::any_cast<YT*>(Y),                           \
                    lambda,                                             \
                    *boost::any_cast<AT*>(A),				\
		    s,							\
		    context);						\
                return;                                                 \
            }                                                           \
        }                                                               \
    }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_FASTERKERNELRIDGE_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mdtypes::dist_matrix_t, mdtypes::dist_matrix_t);

#endif

    SKYLARK_THROW_EXCEPTION (
            base::ml_exception()
            << base::error_msg(
            "FasterKernelRidge has not yet been implemented for this combination of matrices"));
#undef SKYLARK_FASTERKERNELRIDGE_ANY_APPLY_DISPATCH
  
}

} }  // skylark::ml

#endif
