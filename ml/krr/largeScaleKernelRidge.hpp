#ifndef SKYLARK_LAEGE_SCALE_KERNEL_RIDGE_HPP
#define SKYLARK_LAEGE_SCALE_KERNEL_RIDGE_HPP

#ifndef SKYLARK_KRR_HPP
#error "Include top-level krr.hpp instead of including individuals headers"
#endif

#include "KrrParams.hpp"
#include "../kernels/kernels.hpp"

namespace skylark { namespace ml {

template<typename T, typename KernelType>
void LargeScaleKernelRidge(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda,
    bool &scale_maps,
    std::vector<
    sketch::sketch_transform_container_t<El::DistMatrix<T>,
    El::DistMatrix<T> > > &transforms,
    El::DistMatrix<T> &W, El::Int s, base::context_t &context,
    krr_params_t params = krr_params_t()) {

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    boost::mpi::timer timer;

    El::Int t = Y.Width();

    // Create feature transforms.
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Creating feature transforms... ";
        params.log_stream.flush();
        timer.restart();
    }


    El::Int d = direction == base::COLUMNS ? X.Height() : X.Width();

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

        remains -= thiss;
        starts += thiss;
    }

    int C = transforms.size();
    scale_maps = true;

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    // First iteration
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "First iteration (most expensive)... ";
        params.log_stream.flush();
        timer.restart();
    }

    El::Zeros(W, s, t);
    El::DistMatrix<T> R;
    R = Y;

    El::Int s0max = std::max(transforms[0].get_S(), transforms[C - 1].get_S());
    El::DistMatrix<T> Z, ZR;
    if (direction == base::COLUMNS)
        Z.Resize(s0max, X.Width());
    else
        Z.Resize(X.Height(), s0max);
    ZR.Resize(s0max, t);

    std::vector<El::DistMatrix<T> > Ls(C);
    El::DistMatrix<T> W0;
    starts = 0;
    for(int c = 0; c < C; c++) {
        El::Int s0 = transforms[c].get_S();
        El::DistMatrix<T> &L = Ls[c];
        base::RowView(W0, W, starts, s0);

        // Apply feature transform
        if (direction == base::COLUMNS) {
            Z.Resize(s0, X.Width());
            transforms[c].apply(X, Z, sketch::columnwise_tag());
        } else {
            Z.Resize(X.Height(), s0);
            transforms[c].apply(X, Z, sketch::rowwise_tag());
        }

        // Compute factor of local covariance matrix.
        El::Herk(El::LOWER, direction == base::COLUMNS ? El::NORMAL : El::ADJOINT,
            T(1.0), Z, L);
        El::ShiftDiagonal(L, T(lambda));
        El::Cholesky(El::LOWER, L);

        // Compute ZR
        ZR = W0;
        El::Gemm(direction == base::COLUMNS ? El::NORMAL : El::ADJOINT,
            El::NORMAL, T(1.0), Z, R, T(-lambda), ZR);

        // Compute solution, W0
        El::cholesky::SolveAfter(El::LOWER, El::NORMAL, L, ZR);
        El::DistMatrix<T> &delW0 = ZR;
        El::Axpy(T(1.0), delW0, W0);
        El::Gemm(direction == base::COLUMNS ? El::ADJOINT : El::NORMAL,
            El::NORMAL, T(-1.0), Z, delW0, T(1.0), R);

        starts += s0;
    }

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    // More iterations...
    if (log_lev1) {
        params.log_stream << params.prefix
                          << "More iterations... " << std::endl;
        params.log_stream.flush();
        timer.restart();
    }

    for(int it = 1; it < params.iter_lim; it++) {

        starts = 0;
        double delsize = 0.0;

        for(int c = 0; c < C; c++) {
            El::Int s0 = transforms[c].get_S();
            El::DistMatrix<T> &L = Ls[c];
            base::RowView(W0, W, starts, s0);

            // Apply feature transform
            if (direction == base::COLUMNS) {
                Z.Resize(s0, X.Width());
                transforms[c].apply(X, Z, sketch::columnwise_tag());
            } else {
                Z.Resize(X.Height(), s0);
                transforms[c].apply(X, Z, sketch::rowwise_tag());
            }

            // Compute ZR
            ZR = W0;
            El::Gemm(direction == base::COLUMNS ? El::NORMAL : El::ADJOINT,
                El::NORMAL, T(1.0), Z, R, T(-lambda), ZR);

            // Compute solution, W0
            El::cholesky::SolveAfter(El::LOWER, El::NORMAL, L, ZR);
            El::DistMatrix<T> &delW0 = ZR;
            El::Axpy(T(1.0), delW0, W0);
            El::Gemm(direction == base::COLUMNS ? El::ADJOINT : El::NORMAL,
                El::NORMAL, T(-1.0), Z, delW0, T(1.0), R);

            delsize += std::pow(El::FrobeniusNorm(delW0), 2);

            starts += s0;
        }

        double reldel = std::sqrt(delsize) / El::FrobeniusNorm(W);

        if (log_lev2)
            params.log_stream << params.prefix << "\t"
                              << "Iteration " << it
                              << ", relupdate = "
                              << boost::format("%.2e") % reldel
                              << std::endl;

        if (reldel < params.tolerance) {
            if (log_lev2)
                params.log_stream << params.prefix  << "\t"
                                  << "Convergence!" << std::endl;
            break;
        }
    }

    if (log_lev1)
        params.log_stream << params.prefix
                          << "Took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";
}

} }  // skylark::ml

#endif
