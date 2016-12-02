#ifndef SKYLARK_KERNEL_RLSC_HPP
#define SKYLARK_KERNEL_RLSC_HPP

#ifndef SKYLARK_RLSC_HPP
#error "Include top-level rlsc.hpp instead of including individuals headers"
#endif

namespace skylark { namespace ml {

template<typename T, typename R, typename KernelType>
void KernelRLSC(base::direction_t direction, const KernelType &k, 
    const El::DistMatrix<T> &X, const El::DistMatrix<R> &L, T lambda, 
    El::DistMatrix<T> &A, std::vector<R> &rcoding,
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
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec" << std::endl;

    // Solve
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Solving... " << std::endl;
        timer.restart();
    }

    krr_params_t krr_params(params.am_i_printing, params.log_level - 1, 
        params.log_stream, params.prefix + "\t");
    krr_params.use_fast = params.use_fast;
    krr_params.iter_lim = params.iter_lim;
    krr_params.res_print = params.res_print;
    krr_params.tolerance = params.tolerance;

    KernelRidge(base::COLUMNS, k, X, Y, T(lambda), A, krr_params);

    if (log_lev1)
        params.log_stream << params.prefix
                          <<"Solve took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

}

} }  // skylark::ml

#endif