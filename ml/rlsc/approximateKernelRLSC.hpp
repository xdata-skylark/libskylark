#ifndef SKYLARK_APPROXIMATE_RLSC_HPP
#define SKYLARK_APPROXIMATE_RLSC_HPP

#ifndef SKYLARK_RLSC_HPP
#error "Include top-level rlsc.hpp instead of including individuals headers"
#endif

namespace skylark { namespace ml {

template<typename T, typename R, typename KernelType>
void ApproximateKernelRLSC(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<R> &L, T lambda,
    sketch::sketch_transform_container_t<El::DistMatrix<T>, El::DistMatrix<T> > &S,
    El::DistMatrix<T> &W, std::vector<R> &rcoding,
    El::Int s, base::context_t &context,
    rlsc_params_t params = rlsc_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    boost::mpi::timer timer;

    // Form right hand side
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Dummy coding... ";
        params.log_stream.flush();
        timer.restart();
    }

    El::DistMatrix<T> Y;
    std::unordered_map<R, El::Int> coding;
    DummyCoding(El::NORMAL, Y, L, coding, rcoding);

    if (log_lev1)
        params.log_stream  << "took " << boost::format("%.2e") % timer.elapsed()
                           << " sec\n";

    // Solve
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Solving... " << std::endl;
        timer.restart();
    }

    krr_params_t krr_params(params.am_i_printing, params.log_level - 1, 
        params.log_stream, params.prefix + "\t");
    krr_params.use_fast = params.use_fast;
    krr_params.sketched_rr = params.sketched_rls;
    krr_params.sketch_size = params.sketch_size;
    krr_params.fast_sketch = params.fast_sketch;

    ApproximateKernelRidge(direction, k, X, Y,
        T(lambda), S, W, s, context, krr_params);

    if (log_lev1)
        params.log_stream << params.prefix
                          <<"Solve took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";
}


void ApproximateKernelRLSC(base::direction_t direction, const kernel_t &k, 
    const boost::any &X, const boost::any &L, double lambda,
    sketch::sketch_transform_container_t<El::DistMatrix<double>,
        El::DistMatrix<double> > &S, 
    const boost::any &W, std::vector<El::Int> &rcoding,
    El::Int s, base::context_t &context,
    rlsc_params_t params = rlsc_params_t()) {


#define SKYLARK_APPROXIMATEKERNELRLSC_ANY_APPLY_DISPATCH(XT, LT, WT)    \
    if (W.type() == typeid(WT*)) {                                      \
        if (X.type() == typeid(XT*)) {                                  \
            if (L.type() == typeid(LT*)) {                              \
                ApproximateKernelRLSC(direction, k,                     \
                    *boost::any_cast<XT*>(X), *boost::any_cast<LT*>(L), \
                    lambda, S, *boost::any_cast<WT*>(W), rcoding, s,    \
                    context, params);                                   \
                return;                                                 \
            }                                                           \
            if (L.type() == typeid(const LT*)) {                        \
                ApproximateKernelRLSC(direction, k,                     \
                    *boost::any_cast<XT*>(X), *boost::any_cast<LT*>(L), \
                    lambda, S, *boost::any_cast<WT*>(W), rcoding, s,    \
                    context, params);                                   \
                return;                                                 \
            }                                                           \
        }                                                               \
        if (X.type() == typeid(const XT*)) {                            \
            if (L.type() == typeid(LT*)) {                              \
               ApproximateKernelRLSC(direction, k,                      \
                    *boost::any_cast<XT*>(X), *boost::any_cast<LT*>(L), \
                    lambda, S, *boost::any_cast<WT*>(W), rcoding, s,    \
                    context, params);                                   \
                return;                                                 \
            }                                                           \
            if (L.type() == typeid(const LT*)) {                        \
                ApproximateKernelRLSC(direction, k,                     \
                    *boost::any_cast<XT*>(X), *boost::any_cast<LT*>(L), \
                    lambda, S, *boost::any_cast<WT*>(W), rcoding, s,    \
                    context, params);                                   \
                return;                                                 \
            }                                                           \
        }                                                               \
    }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_APPROXIMATEKERNELRLSC_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
         mitypes::dist_matrix_t,  mdtypes::dist_matrix_t);

    SKYLARK_APPROXIMATEKERNELRLSC_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
         mitypes::dist_matrix_t, mdtypes::dist_matrix_t);

#endif

    SKYLARK_THROW_EXCEPTION (
            base::ml_exception()
            << base::error_msg(
            "ApproximateKernelRLSC has not yet been implemented for this"
            "combination of matrices"));

#undef SKYLARK_APPROXIMATEKERNELRLSC_ANY_APPLY_DISPATCH
}

} }  // skylark::ml

#endif