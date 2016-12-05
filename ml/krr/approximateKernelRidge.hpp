#ifndef SKYLARK_APPROXIMATE_KERNEL_RIDGE_HPP
#define SKYLARK_APPROXIMATE_KERNEL_RIDGE_HPP

#ifndef SKYLARK_KRR_HPP
#error "Include top-level krr.hpp instead of including individuals headers"
#endif

#include "KrrParams.hpp"
#include "../kernels/kernels.hpp"

namespace skylark { namespace ml {

// TODO(Jordi): Should we mix the logic with logging?
template<typename T, typename KernelType>
void ApproximateKernelRidge(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda,
    sketch::sketch_transform_container_t<El::DistMatrix<T>, El::DistMatrix<T> > &S,
    El::DistMatrix<T> &W, El::Int s, base::context_t &context,
    krr_params_t params = krr_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    boost::mpi::timer timer;

    // Create and apply the feature transform
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Create and apply the feature transform... ";
        params.log_stream.flush();
        timer.restart();
   }

    sketch::generic_sketch_transform_t *p0 = params.use_fast ?
        k.create_rft(s, fast_feature_transform_tag(), context) :
        k.create_rft(s, regular_feature_transform_tag(), context);
    sketch::generic_sketch_transform_ptr_t p(p0);
    S =
        sketch::sketch_transform_container_t<El::DistMatrix<T>,
                                             El::DistMatrix<T> >(p);

    El::DistMatrix<T> Z;

    if (direction == base::COLUMNS) {
        Z.Resize(s, X.Width());
        S.apply(X, Z, sketch::columnwise_tag());
    } else {
        Z.Resize(X.Height(), s);
        S.apply(X, Z, sketch::rowwise_tag());
    }

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    // Sketch the problem (if requested)
    El::DistMatrix<T> SZ, SY;
    if (params.sketched_rr) {

        if (log_lev1) {
            params.log_stream << params.prefix
                              << "Sketching the regression problem... ";
            params.log_stream.flush();
            timer.restart();
        }

        El::Int m = direction ==  base::COLUMNS ? Z.Width() : Z.Height();
        El::Int t = params.sketch_size == -1 ? 4 * s : params.sketch_size;

        sketch::sketch_transform_t<El::DistMatrix<T>, El::DistMatrix<T> > *R;
        if (params.fast_sketch)
            R = new sketch::CWT_t<El::DistMatrix<T>,
                                  El::DistMatrix<T> >(m, t, context);
        else
            R = new sketch::FJLT_t<El::DistMatrix<T>,
                                   El::DistMatrix<T> >(m, t, context);

        if (direction == base::COLUMNS) {
            SZ.Resize(s, t);
            R->apply(Z, SZ, sketch::rowwise_tag());

            // TODO it is "wrong" that Y is oriented differently than X/Z
            SY.Resize(t, Y.Width());
            R->apply(Y, SY, sketch::columnwise_tag());

        } else {
            SZ.Resize(t, s);
            R->apply(Z, SZ, sketch::columnwise_tag());
            SY.Resize(t, Y.Width());
            R->apply(Y, SY, sketch::columnwise_tag());
        }

        delete R;

        if (log_lev1)
            params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                              << " sec\n";
    } else {
        El::View(SZ, Z);
        El::LockedView(SY, Y);
    }

    // Solving the regression problem
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Solving the regression problem... ";
        params.log_stream.flush();
            timer.restart();
    }

    El::Ridge(direction == base::COLUMNS ? El::ADJOINT : El::NORMAL,
        SZ, SY, std::sqrt(lambda), W);

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";
}

void ApproximateKernelRidge(base::direction_t direction, const kernel_t &k,
    const  boost::any &X, const  boost::any &Y, double lambda,
    sketch::sketch_transform_container_t<El::DistMatrix<double>,
        El::DistMatrix<double> > &S,
    const boost::any &W, El::Int s, base::context_t &context,
    krr_params_t params = krr_params_t()) {


#define SKYLARK_APPROXIMATEKERNELRIDGE_ANY_APPLY_DISPATCH(XT, YT, WT)   \
    if (W.type() == typeid(WT*)) {                                      \
        if (X.type() == typeid(XT*)) {                                  \
            if (Y.type() == typeid(YT*)) {                              \
                ApproximateKernelRidge(direction, k,                    \
                    *boost::any_cast<XT*>(X),*boost::any_cast<YT*>(Y),  \
                    lambda, S, *boost::any_cast<WT*>(W), s,             \
                    context, params);                                   \
                return;                                                 \
            }                                                           \
            if (Y.type() == typeid(const YT*)) {                        \
                ApproximateKernelRidge(direction, k,                    \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, S,  *boost::any_cast<WT*>(W), s,            \
                    context, params);                                   \
                return;                                                 \
            }                                                           \
        }                                                               \
        if (X.type() == typeid(const XT*)) {                            \
            if (Y.type() == typeid(YT*)) {                              \
                ApproximateKernelRidge(direction, k,                    \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, S, *boost::any_cast<WT*>(W), s,             \
                    context, params);                                          \
                return;                                                 \
            }                                                           \
                                                                        \
            if (Y.type() == typeid(const YT*)) {                        \
                ApproximateKernelRidge(direction, k,                    \
                    *boost::any_cast<XT*>(X), *boost::any_cast<YT*>(Y), \
                    lambda, S, *boost::any_cast<WT*>(W), s,             \
                    context, params);                                          \
                return;                                                 \
            }                                                           \
        }                                                               \
    }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_APPROXIMATEKERNELRIDGE_ANY_APPLY_DISPATCH(mdtypes::dist_matrix_t,
        mdtypes::dist_matrix_t, mdtypes::dist_matrix_t);

    SKYLARK_APPROXIMATEKERNELRIDGE_ANY_APPLY_DISPATCH(mftypes::dist_matrix_t,
        mftypes::dist_matrix_t, mftypes::dist_matrix_t);
#endif

    SKYLARK_THROW_EXCEPTION (
            base::ml_exception()
            << base::error_msg(
            "ApproximateKernelRidge has not yet been implemented for this combination"
            "of matrices"));
#undef SKYLARK_APPROXIMATEKERNELRIDGE_ANY_APPLY_DISPATCH

}

} }  // skylark::ml

#endif
