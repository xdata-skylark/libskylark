#ifndef SKYLARK_KRR_HPP
#define SKYLARK_KRR_HPP

namespace skylark { namespace ml {

struct krr_params_t : public base::params_t {

    // For iterative methods (FasterKRR)
    int iter_lim;
    int res_print;
    double tolerance;

    krr_params_t(bool am_i_printing = 0,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        std::string prefix = "", 
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, prefix, debug_level) {

        tolerance = 1e-3;
        res_print = 10;
        iter_lim = 1000;
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


template<typename MatrixType>
class feature_map_precond_t :
    public algorithms::outplace_precond_t<MatrixType, MatrixType> {

public:

    typedef MatrixType matrix_type;
    typedef typename utility::typer_t<matrix_type>::value_type value_type;

    virtual bool is_id() const { return false; }

    template<typename KernelType, typename InputType>
    feature_map_precond_t(const KernelType &k, value_type lambda,
        const InputType &X, El::Int s, base::context_t &context) {
        _lambda = lambda;
        _s = s;

        U.Resize(s, X.Width());
        sketch::sketch_transform_t<InputType, matrix_type> *S =
            k.template create_rft<InputType, matrix_type>(s,
                ml::regular_feature_transform_tag(),
                context);
        S->apply(X, U, sketch::columnwise_tag());
        delete S;

        El::Identity(C, s, s);

        El::Herk(El::LOWER, El::NORMAL, value_type(1.0)/_lambda, U,
            value_type(1.0), C);
        El::Cholesky(El::LOWER, C);

        // El::SymmetricInverse(El::LOWER, C);

        //El::Gemm(El::NORMAL, El::ADJOINT, 1.0/_lambda, U, U, 1.0, C);
        //El::Inverse(C);
    }

    virtual void apply(const matrix_type& B, matrix_type& X) const {
        //matrix_type UB(_s, B.Width()), CUB(_s, B.Width());
        //El::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), U, B, UB);
        //El::Hemm(El::LEFT, El::LOWER, value_type(1.0), C, UB,
        //    value_type(0.0), CUB);

        matrix_type CUB(_s, B.Width());
        El::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), U, B, CUB);
        El::cholesky::SolveAfter(El::LOWER, El::NORMAL, C, CUB);

        X = B;
        El::Gemm(El::ADJOINT, El::NORMAL, value_type(-1.0) / (_lambda * _lambda), 
            U, CUB, value_type(1.0)/_lambda, X);
    }

    virtual void apply_adjoint(const matrix_type& B, matrix_type& X) const {
        apply(B, X);
    }

private:
    value_type _lambda;
    El::Int _s;
    matrix_type U, C;
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
                          << "Creating precoditioner... ";
        params.log_stream.flush();
        timer.restart();
    }

    feature_map_precond_t<El::DistMatrix<T> > P(k, lambda, X, s, context);

    if (log_lev1)
        params.log_stream << "took " << boost::format("%.2e") % timer.elapsed()
                          << " sec\n";

    if (log_lev1) {
        params.log_stream << params.prefix
                          << "Solving linear equation... "
                          << std::endl;
        params.log_stream.flush();
        timer.restart();
    }


    // Solve
    algorithms::krylov_iter_params_t cg_params;
    cg_params.iter_lim = params.iter_lim;
    cg_params.res_print = params.res_print;
    cg_params.log_level = params.log_level;
    cg_params.am_i_printing = params.am_i_printing;
    cg_params.prefix = params.prefix + "\t";
    cg_params.tolerance = params.tolerance;

    El::Zeros(A, X.Width(), Y.Width());
    algorithms::CG(El::LOWER, K, Y, A, cg_params, P);

    if (log_lev1)
        params.log_stream  << params.prefix
                           <<"Took " << boost::format("%.2e") % timer.elapsed()
                           << " sec\n";

}

} } // namespace skylark::ml

#endif
