#ifndef SKYLARK_KERNEL_RIDGE_HPP
#define SKYLARK_KERNEL_RIDGE_HPP

#ifndef SKYLARK_KRR_HPP
#error "Include top-level krr.hpp instead of including individuals headers"
#endif

#include "KrrParams.hpp"
#include "../kernels/kernels.hpp"

namespace skylark { namespace ml {

// TODO(Jordi): Should we mix the logic with logging?
template<typename T, typename KernelType>
void KernelRidge(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda,
    El::DistMatrix<T> &A, krr_params_t params = krr_params_t()) {

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

    El::DistMatrix<T> K;
    SymmetricGram(El::LOWER, direction, k, X, K);

    // Add regularizer
    El::DistMatrix<T> D;
    El::Ones(D, X.Width(), 1);
    El::UpdateDiagonal(K, lambda, D);

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Solving the equation... ";
        params.log_stream.flush();
        timer.restart();
    }

    A = Y;
    HPDSolve(El::LOWER, El::NORMAL, K, A);

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";
}

void KernelRidge(base::direction_t direction, const kernel_t &k,
    const boost::any &X, const boost::any &Y, double lambda,
    const boost::any &A, krr_params_t params = krr_params_t()) {

#define SKYLARK_KERNELRIDGE_ANY_APPLY_DISPATCH(XT, YT, AT)              \
    if (A.type() == typeid(AT*)) {                                      \
        if (X.type() == typeid(XT*)) {                                  \
            if (Y.type() == typeid(YT*)) {                              \
                KernelRidge(direction, k,                               \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, *boost::any_cast<AT*>(A), params);          \
                return;                                                 \
            }                                                           \
            if (Y.type() == typeid(const YT*)) {                        \
                KernelRidge(direction, k,                               \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, *boost::any_cast<AT*>(A), params);          \
                return;                                                 \
            }                                                           \
        }                                                               \
        if (X.type() == typeid(const XT*)) {                            \
            if (Y.type() == typeid(YT*)) {                              \
                KernelRidge(direction, k,                               \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, *boost::any_cast<AT*>(A), params);          \
                return;                                                 \
            }                                                           \
                                                                        \
            if (Y.type() == typeid(const YT*)) {                        \
                KernelRidge(direction, k,                               \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, *boost::any_cast<AT*>(A), params);          \
                return;                                                 \
            }                                                           \
        }                                                               \
    }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_KERNELRIDGE_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mdtypes::dist_matrix_t, mdtypes::dist_matrix_t);

    SKYLARK_KERNELRIDGE_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
        mftypes::dist_matrix_t, mftypes::dist_matrix_t);

#endif
}

} }  // skylark::ml

#endif
