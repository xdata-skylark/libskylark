#ifndef SKYLARK_SKETCHED_APPROXIMATE_RIDGE_HPP
#define SKYLARK_SKETCHED_APPROXIMATE_RIDGE_HPP

#ifndef SKYLARK_KRR_HPP
#error "Include top-level krr.hpp instead of including individuals headers"
#endif

#include "KrrParams.hpp"
#include "../kernels/kernels.hpp"

namespace skylark { namespace ml {

template<typename T, typename KernelType>
void SketchedApproximateKernelRidge(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda,
    bool &scale_maps,
    std::vector<
    sketch::sketch_transform_container_t<El::DistMatrix<T>, 
    El::DistMatrix<T> > > &transforms,
    El::DistMatrix<T> &W, El::Int s, El::Int t, base::context_t &context,
    krr_params_t params = krr_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    boost::mpi::timer timer;

    // Both sketches
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Feature transforms + Sketch... ";
        params.log_stream.flush();
        timer.restart();
    }

    El::DistMatrix<T> Z, RZ, SZ, SY, VSZ;

    El::Int d = direction == base::COLUMNS ? X.Height() : X.Width();


    El::Int m = direction ==  base::COLUMNS ? X.Width() : X.Height();
    t = t == -1 ? 4 * s : t;

    if (direction == base::COLUMNS)
        SZ.Resize(s, t);
    else
        SZ.Resize(t, s);

    sketch::sketch_transform_t<El::DistMatrix<T>, El::DistMatrix<T> > *R;
    if (params.fast_sketch)
        R = new sketch::CWT_t<El::DistMatrix<T>,
                              El::DistMatrix<T> >(m, t, context);
    else
        R = new sketch::FJLT_t<El::DistMatrix<T>,
                               El::DistMatrix<T> >(m, t, context);


    SY.Resize(t, 1);
    R->apply(Y, SY, sketch::columnwise_tag());

    El::Int starts = 0, remains = s;
    El::Int sinc = (params.max_split == 0) ? d : params.max_split / 2;

    transforms.resize(0);
    transforms.reserve((s / sinc) + 1);

    while (remains > 0) {
        El::Int thiss = (remains <= 2 * sinc) ? remains : sinc;
        sketch::generic_sketch_transform_t *p0 = params.use_fast ?
            k.create_rft(thiss, fast_feature_transform_tag(), context) :
            k.create_rft(thiss, regular_feature_transform_tag(), context);
        sketch::generic_sketch_transform_ptr_t p(p0);
        sketch::sketch_transform_container_t<El::DistMatrix<T>,
                                             El::DistMatrix<T> > S(p);
        transforms.push_back(S);

        if (direction == base::COLUMNS) {

            Z.Resize(thiss, X.Width());
            S.apply(X, Z, sketch::columnwise_tag());
            El::Scale<T,T>(sqrt(T(thiss) / s), Z);
            RZ.Resize(thiss, t);
            R->apply(Z, RZ, sketch::rowwise_tag());
            base::RowView(VSZ, SZ, starts, thiss);
            VSZ = RZ;

        } else {
            Z.Resize(X.Height(), thiss);
            S.apply(X, Z, sketch::rowwise_tag());
            El::Scale<T,T>(sqrt(T(thiss) / s), Z);
            RZ.Resize(t, thiss);
            R->apply(Z, RZ, sketch::columnwise_tag());
            base::ColumnView(VSZ, SZ, starts, thiss);
            VSZ = RZ;

        }

        remains -= thiss;
        starts += thiss;
    }

    delete R;

    scale_maps = true;

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

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

} }  // skylark::ml

#endif
