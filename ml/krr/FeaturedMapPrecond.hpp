#ifndef SKYLARK_FEATURED_MAP_PRECOND_HPP
#define SKYLARK_FEATURED_MAP_PRECOND_HPP

#ifndef SKYLARK_KRR_HPP
#error "Include top-level krr.hpp instead of including individuals headers"
#endif

#include "KrrParams.hpp"
#include "../kernels/kernels.hpp"

namespace skylark { namespace ml {

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

        _lambda = params.precond_lambda == -1 ? lambda :
            params.precond_lambda;;
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

} }  // skylark::ml

#endif
