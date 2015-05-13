#ifndef SKYLARK_KRR_HPP
#define SKYLARK_KRR_HPP

namespace skylark { namespace ml {

template<typename T, typename KernelType>
void KernelRidge(base::direction_t direction, const KernelType &k, 
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda, 
    El::DistMatrix<T> &A) {

    // TODO: Temporary!
    boost::mpi::communicator world;
    int rank = world.rank();

    boost::mpi::timer timer;

    // Compute kernel matrix
    if (rank == 0) {
        std::cout << "\tComputing kernel matrix... ";
        std::cout.flush();
        timer.restart();
    }

    El::DistMatrix<double> K;
    SymmetricGram(El::LOWER, direction, k, X, K);

    // Add regularizer
    El::DistMatrix<double> D;
    El::Ones(D, X.Width(), 1);
    El::UpdateDiagonal(K, lambda, D);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (rank == 0) {
        std::cout << "\tSolving the equation... ";
        std::cout.flush();
        timer.restart();
    }

    A = Y;
    HPDSolve(El::LOWER, El::NORMAL, K, A);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

}


template<typename MatrixType>
class feature_map_precond_t :
    public skylark::algorithms::outplace_precond_t<MatrixType, MatrixType> {

public:

    typedef MatrixType matrix_type;

    virtual bool is_id() const { return false; }

    template<typename KernelType, typename InputType>
    feature_map_precond_t(const KernelType &k, double lambda,
        const InputType &X, El::Int s, skylark::base::context_t &context) {
        _lambda = lambda;
        _s = s;

        U.Resize(s, X.Width());
        skylark::sketch::sketch_transform_t<InputType, matrix_type> *S =
            k.template create_rft<InputType, matrix_type>(s,
                skylark::ml::regular_feature_transform_tag(),
                context);
        S->apply(X, U, skylark::sketch::columnwise_tag());
        delete S;

        El::Identity(C, s, s);

        El::Herk(El::LOWER, El::NORMAL, 1.0/_lambda, U, 1.0, C);
        El::SymmetricInverse(El::LOWER, C);

        //El::Gemm(El::NORMAL, El::ADJOINT, 1.0/_lambda, U, U, 1.0, C);
        //El::Inverse(C);
    }

    virtual void apply(const matrix_type& B, matrix_type& X) const {
        matrix_type UB(_s, B.Width()), CUB(_s, B.Width());
        El::Gemm(El::NORMAL, El::NORMAL, 1.0, U, B, 0.0, UB);
        El::Hemm(El::LEFT, El::LOWER, 1.0, C, UB, 0.0, CUB);

        X = B;
        El::Gemm(El::ADJOINT, El::NORMAL, -1.0 / (_lambda * _lambda), U, CUB,
            1.0/_lambda, X);
    }

    virtual void apply_adjoint(const matrix_type& B, matrix_type& X) const {
        apply(B, X);
    }

private:
    double _lambda;
    El::Int _s;
    matrix_type U, C;
};

template<typename T, typename KernelType>
void FasterKernelRidge(base::direction_t direction, const KernelType &k, 
    const El::DistMatrix<T> &X, const El::DistMatrix<T> &Y, T lambda, 
    El::DistMatrix<T> &A, El::Int s, base::context_t &context) {

    // TODO: Temporary!
    boost::mpi::communicator world;
    int rank = world.rank();

    boost::mpi::timer timer;

    // Compute kernel matrix
    if (rank == 0) {
        std::cout << "\tComputing kernel matrix... ";
        std::cout.flush();
        timer.restart();
    }

    El::DistMatrix<double> K;
    SymmetricGram(El::LOWER, direction, k, X, K);

    // Add regularizer
    El::DistMatrix<double> D;
    El::Ones(D, X.Width(), 1);
    El::UpdateDiagonal(K, lambda, D);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (rank == 0) {
        std::cout << "\tCreating precoditioner... ";
        std::cout.flush();
        timer.restart();
    }

    feature_map_precond_t<El::DistMatrix<double> > P(k, lambda, X, s, context);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (rank == 0) {
        std::cout << "\tSolving linear equation... ";
        std::cout.flush();
        timer.restart();
    }


    // Solve
    algorithms::krylov_iter_params_t cg_params;
    cg_params.iter_lim = 1000;
    cg_params.res_print = 10;
    cg_params.log_level = 2;
    cg_params.am_i_printing = rank == 0;
    cg_params.tolerance = 1e-3;

    El::Zeros(A, X.Width(), Y.Width());
    algorithms::CG(El::LOWER, K, Y, A, cg_params, P);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

}

} } // namespace skylark::ml

#endif 
