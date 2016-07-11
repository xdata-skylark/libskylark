#ifndef SKYLARK_KRR_HPP
#define SKYLARK_KRR_HPP

#include "../utility/timer.hpp"

namespace skylark { namespace ml {

struct krr_params_t : public base::params_t {

    // For all methods that use feature transforms
    bool use_fast;

    // For approximate methods (ApproximateKRR)
    bool sketched_rr;
    El::Int sketch_size;
    bool fast_sketch;

    // For iterative methods (FasterKRR, LargeScaleKRR)
    int iter_lim;
    int res_print;
    double tolerance;

    // For memory limited methods (SketchedApproximateKRR, LargeScaleKRR)
    El::Int max_split;

    krr_params_t(bool am_i_printing = 0,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        std::string prefix = "",
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, prefix, debug_level) {

        use_fast = false;

        sketched_rr = false;
        sketch_size = -1;
        fast_sketch = false;

        tolerance = 1e-3;
        res_print = 10;
        iter_lim = 1000;

        max_split = 0;
  }

};

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

template<typename MatrixType>
class feature_map_precond_t :
    public algorithms::outplace_precond_t<MatrixType, MatrixType> {

public:

    typedef MatrixType matrix_type;
    typedef typename utility::typer_t<matrix_type>::value_type value_type;

    virtual bool is_id() const { return false; }

    template<typename KernelType, typename InputType>
    feature_map_precond_t(const KernelType &k, value_type lambda,
        const InputType &X, El::Int s, base::context_t &context,
        const krr_params_t &params) {

        SKYLARK_TIMER_DINIT(KRR_PRECOND_GEMM1_PROFILE);
        SKYLARK_TIMER_DINIT(KRR_PRECOND_GEMM2_PROFILE);
        SKYLARK_TIMER_DINIT(KRR_PRECOND_COPY_PROFILE);

        _lambda = lambda;
        _s = s;

        bool log_lev2 = params.am_i_printing && params.log_level >= 2;

        boost::mpi::timer timer;

        if (log_lev2) {
            params.log_stream << params.prefix << "\t"
                              << "Applying random features transform... ";
            params.log_stream.flush();
            timer.restart();
        }

        U.Resize(s, X.Width());
        sketch::sketch_transform_t<InputType, matrix_type> *S =
            params.use_fast ? 
            k.template create_rft<InputType, matrix_type>(s,
                ml::fast_feature_transform_tag(),
                context)
            :
            k.template create_rft<InputType, matrix_type>(s,
                ml::regular_feature_transform_tag(),
                context);
        S->apply(X, U, sketch::columnwise_tag());
        delete S;

        if (log_lev2)
            params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                              << " sec\n";

        if (log_lev2) {
            params.log_stream << params.prefix << "\t"
                              << "Computing covariance matrix... ";
            params.log_stream.flush();
            timer.restart();
        }

        matrix_type C;
        El::Identity(C, s, s);
        El::Herk(El::LOWER, El::NORMAL, value_type(1.0)/_lambda, U,
            value_type(1.0), C);

        if (log_lev2)
            params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                              << " sec\n";

        if (log_lev2) {
            params.log_stream << params.prefix << "\t"
                              << "Factorizing... ";
            params.log_stream.flush();
            timer.restart();
        }

        El::Cholesky(El::LOWER, C);

        if (log_lev2)
            params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                              << " sec\n";

        if (log_lev2) {
            params.log_stream << params.prefix << "\t"
                              << "Prepare factor... ";
            params.log_stream.flush();
            timer.restart();
        }

        El::Trsm(El::LEFT, El::LOWER, El::NORMAL, El::NON_UNIT,
            value_type(1.0)/_lambda, C, U);

        if (log_lev2)
            params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                              << " sec\n";

    }

    virtual ~feature_map_precond_t() {
        auto comm = utility::get_communicator(U);
        SKYLARK_TIMER_PRINT(KRR_PRECOND_GEMM1_PROFILE, comm);
        SKYLARK_TIMER_PRINT(KRR_PRECOND_GEMM2_PROFILE, comm);
        SKYLARK_TIMER_PRINT(KRR_PRECOND_COPY_PROFILE, comm);
    }

    virtual void apply(const matrix_type& B, matrix_type& X) const {

        matrix_type CUB(_s, B.Width());

        // Really makes sense to keep U not communicated...
        // Bypass Elemental defaults since they seem to be generating bad
        // choices for larger matrices.

        SKYLARK_TIMER_RESTART(KRR_PRECOND_GEMM1_PROFILE);
        El::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), U, B, CUB,
            El::GEMM_SUMMA_A);
        SKYLARK_TIMER_ACCUMULATE(KRR_PRECOND_GEMM1_PROFILE);

        SKYLARK_TIMER_RESTART(KRR_PRECOND_COPY_PROFILE);
        X = B;
        SKYLARK_TIMER_ACCUMULATE(KRR_PRECOND_COPY_PROFILE);

        SKYLARK_TIMER_RESTART(KRR_PRECOND_GEMM2_PROFILE);
        El::Gemm(El::ADJOINT, El::NORMAL, value_type(-1.0),
            U, CUB, value_type(1.0)/_lambda, X);
        SKYLARK_TIMER_ACCUMULATE(KRR_PRECOND_GEMM2_PROFILE);
    }

    virtual void apply_adjoint(const matrix_type& B, matrix_type& X) const {
        apply(B, X);
    }

private:
    value_type _lambda;
    El::Int _s;
    matrix_type U;

    SKYLARK_TIMER_DECLARE(KRR_PRECOND_GEMM1_PROFILE);
    SKYLARK_TIMER_DECLARE(KRR_PRECOND_GEMM2_PROFILE);
    SKYLARK_TIMER_DECLARE(KRR_PRECOND_COPY_PROFILE);
};

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

} } // namespace skylark::ml

#endif
