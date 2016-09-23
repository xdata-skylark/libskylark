#ifndef SKYLARK_ACCELERATED_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP
#define SKYLARK_ACCELERATED_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP

#include <El.hpp>

#include "regression_problem.hpp"

namespace skylark {
namespace algorithms {

namespace flinl2_internal {

extern "C" {

void EL_BLAS(dtrcon)(const char* norm, const char* uplo, const char* diag,
    const int* n, const double* A, const int *lda, double *rcond,
    double *work, int *iwork, int *info);

void EL_BLAS(strcon)(const char* norm, const char* uplo, const char* diag,
    const int* n, const float* A, const int *lda, float *rcond,
    float *work, int *iwork, int *info);

}

inline double utcondest(const El::Matrix<double>& A) {
    int n = A.Height(), ld = A.LDim(), info;
    double *work = new double[3 * n];
    int *iwork = new int[n];
    double rcond;
    EL_BLAS(dtrcon)("1", "U", "N", &n, A.LockedBuffer(), &ld, &rcond, work,
        iwork, &info); // TODO check info
    delete[] work;
    delete[] iwork;
    return 1/rcond;
}

inline double utcondest(const El::Matrix<float>& A) {
    int n = A.Height(), ld = A.LDim(), info;
    float *work  = new float[3 * n];
    int *iwork = new int[n];
    float rcond;
    EL_BLAS(strcon)("1", "U", "N", &n, A.LockedBuffer(), &ld, &rcond, work,
        iwork, &info);  // TODO check info
    delete[] work;
    delete[] iwork;
    return 1/rcond;
}

template<typename T>
inline double utcondest(const El::DistMatrix<T, El::STAR, El::STAR>& A) {
    return utcondest(A.LockedMatrix());
}

template<typename T>
inline double utcondest(const El::DistMatrix<T, El::CIRC, El::CIRC>& A) {
    return utcondest(A.LockedMatrix());
}

template<typename T>
inline double utcondest(const El::DistMatrix<T>& A) {
    // Probably slower than condition number estimation in LAPACK
    El::DistMatrix<T> invA(A);
    El::TriangularInverse(El::UPPER, El::NON_UNIT, invA);
    return El::OneNorm(A) * El::OneNorm(invA);
}

template<typename SolType, typename SketchType, typename PrecondType>
double build_precond(SketchType& SA,
    PrecondType& R, algorithms::inplace_precond_t<SolType> *&P, qr_precond_tag) {
    El::qr::Explicit(SA, R);
    P =
        new algorithms::inplace_tri_inverse_precond_t<SolType, PrecondType,
                                       El::UPPER, El::NON_UNIT>(R);

    return utcondest(R);
}

template<typename SolType, typename SketchType, typename PrecondType>
double build_precond(SketchType& SA,
    PrecondType& V, algorithms::inplace_precond_t<SolType> *&P, svd_precond_tag) {

    int n = SA.Width();
    PrecondType s(SA); // TODO should s be PrecondType or STAR,STAR ?
    s.Resize(n, 1);
    El::SVD(SA, SA, s, V);
    for(int i = 0; i < n; i++)
        s.Set(i, 0, 1 / s.Get(i, 0));
    El::DiagonalScale(El::RIGHT, El::NORMAL, s, V);
    P =
        new algorithms::inplace_mat_precond_t<SolType, PrecondType>(V);
    return s.Get(0,0) / s.Get(n-1, 0);
}

}  // namespace flinl2_internal

/// Specialization for simplified Blendenpik algorithm
template <typename ValueType, El::Distribution VD,
          template <typename, typename> class TransformType,
          typename PrecondTag>
class accelerated_regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, VD, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    simplified_blendenpik_tag<TransformType, PrecondTag> > {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, VD, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> precond_type;
    typedef precond_type sketch_type;
    // The assumption is that the sketch is not much bigger than the
    // preconditioner, so we should use the same matrix distribution.

    const int _m;
    const int _n;
    const matrix_type &_A;
    precond_type _R;
    algorithms::inplace_precond_t<sol_type> *_precond_R;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     * @param context Skylark context.
     */
    accelerated_regression_solver_t(const problem_type& problem,
        base::context_t& context) :
        _m(problem.m), _n(problem.n), _A(problem.input_matrix),
        _R(_n, _n, problem.input_matrix.Grid()) {
        // TODO n < m ???

        int t = 4 * _n;    // TODO parameter.

        TransformType<matrix_type, sketch_type> S(_m, t, context);
        sketch_type SA(t, _n);
        S.apply(_A, SA, sketch::columnwise_tag());

        flinl2_internal::build_precond(SA, _R, _precond_R, PrecondTag());
    }

    ~accelerated_regression_solver_t() {
        delete _precond_R;
    }

    int solve(const rhs_type& b, sol_type& x) {
        return LSQR(_A, b, x, algorithms::krylov_iter_params_t(), *_precond_R);
    }
};

/**
 * Specialization: Blendenpik, [VC/VR,STAR] input, [STAR, STAR] solution.
 */
template <typename ValueType, El::Distribution VD,
          typename PrecondTag>
class accelerated_regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, VD, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    blendenpik_tag<PrecondTag> > {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, VD, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> precond_type;
    typedef precond_type sketch_type;
    // The assumption is that the sketch is not much bigger than the
    // preconditioner, so we should use the same matrix distribution.

    const int _m;
    const int _n;
    const matrix_type &_A;
    precond_type _R;
    algorithms::inplace_precond_t<sol_type> *_precond_R;

    regression_solver_t<problem_type, rhs_type, sol_type, svd_l2_solver_tag>
    *_alt_solver;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     * @param context Skylark context.
     */
    accelerated_regression_solver_t(const problem_type& problem, base::context_t& context) :
        _m(problem.m), _n(problem.n), _A(problem.input_matrix),
        _R(_n, _n, problem.input_matrix.Grid()) {
        // TODO n < m ???

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF
        int t = 4 * _n;    // TODO parameter.
        double scale = std::sqrt((double)_m / (double)t);

        El::DistMatrix<ValueType, El::STAR, VD> Ar(_A.Grid());
        El::DistMatrix<ValueType, El::STAR, VD> dist_SA(t, _n, _A.Grid());
        sketch_type SA(t, _n, _A.Grid());
        boost::random::uniform_int_distribution<int> distribution(0, _m- 1);

        Ar = _A;
        double condest = 0;
        int attempts = 0;
        do {
            sketch::RFUT_t<El::DistMatrix<ValueType, El::STAR, VD>,
                           sketch::fft_futs<double>::DCT_t,
                           utility::rademacher_distribution_t<value_type> >
                F(_m, context);
            F.apply(Ar, Ar, sketch::columnwise_tag());

            std::vector<int> samples =
                context.generate_random_samples_array(t, distribution);
            for (int j = 0; j < Ar.LocalWidth(); j++)
                for (int i = 0; i < t; i++) {
                    int row = samples[i];
                    dist_SA.Matrix().Set(i, j, scale * Ar.Matrix().Get(row, j));
                }

            SA = dist_SA;
            condest = flinl2_internal::build_precond(SA, _R, _precond_R, PrecondTag());
            attempts++;
        } while (condest > 1e14 && attempts < 3); // TODO parameters

        if (condest <= 1e14)
            _alt_solver = nullptr;
        else {
            _alt_solver =
                new regression_solver_t<problem_type,
                                      rhs_type,
                                      sol_type, svd_l2_solver_tag>(problem);
            delete _precond_R;
            _precond_R = nullptr;
        }
#else
        //TODO: how to handle?
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Requires FFT support!"));
#endif
    }

    ~accelerated_regression_solver_t() {
        if (_precond_R != nullptr)
            delete _precond_R;
        if (_alt_solver != nullptr)
            delete _alt_solver;
    }

    int solve(const rhs_type& b, sol_type& x) {
        if (_precond_R != nullptr)
            return LSQR(_A, b, x, algorithms::krylov_iter_params_t(),
                *_precond_R);
        else {
            _alt_solver->solve(b, x);
            return 0;
        }
    }
};

/**
 * Specialization: Blendenpik, [MC, MR] input, [MC, MR] solution.
 */
template <typename ValueType, El::Distribution U, El::Distribution V,
          typename PrecondTag>
class accelerated_regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, U, V>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType>,
    El::DistMatrix<ValueType>,
    blendenpik_tag<PrecondTag> > {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType> matrix_type;
    typedef El::DistMatrix<ValueType> rhs_type;
    typedef El::DistMatrix<ValueType> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef El::DistMatrix<ValueType> precond_type;
    typedef precond_type sketch_type;
    // The assumption is that the sketch is not much bigger than the
    // preconditioner, so we should use the same matrix distribution.

    const int _m;
    const int _n;
    const matrix_type &_A;
    precond_type _R;
    algorithms::inplace_precond_t<sol_type> *_precond_R;

    regression_solver_t<problem_type, rhs_type, sol_type, svd_l2_solver_tag>
    *_alt_solver;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     * @param context Skylark context.
     */
    accelerated_regression_solver_t(const problem_type& problem,
            base::context_t& context) :
        _m(problem.m), _n(problem.n), _A(problem.input_matrix),
        _R(_n, _n, problem.input_matrix.Grid()) {
        // TODO n < m ???

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF
        int t = 4 * _n;    // TODO parameter.
        double scale = std::sqrt((double)_m / (double)t);

        El::DistMatrix<ValueType, El::STAR, El::VR> Ar(_A.Grid());
        El::DistMatrix<ValueType, El::STAR, El::VR> dist_SA(t, _n, _A.Grid());
        sketch_type SA(t, _n, _A.Grid());
        boost::random::uniform_int_distribution<int> distribution(0, _m- 1);

        Ar = _A;
        double condest = 0;
        int attempts = 0;
        do {
            sketch::RFUT_t<El::DistMatrix<value_type, El::STAR, El::VR>,
                           typename sketch::fft_futs<value_type>::DCT_t,
                           utility::rademacher_distribution_t<double> >
                F(_m, context);
            F.apply(Ar, Ar, sketch::columnwise_tag());

            std::vector<int> samples =
                context.generate_random_samples_array(t, distribution);
            for (int j = 0; j < Ar.LocalWidth(); j++)
                for (int i = 0; i < t; i++) {
                    int row = samples[i];
                    dist_SA.Matrix().Set(i, j, scale * Ar.Matrix().Get(row, j));
                }

            SA = dist_SA;
            condest = flinl2_internal::build_precond(SA, _R, _precond_R,
                PrecondTag());
            attempts++;
        } while (condest > 1e14 && attempts < 3); // TODO parameters

        if (condest <= 1e14)
            _alt_solver = nullptr;
        else {
            _alt_solver =
                new regression_solver_t<problem_type,
                                      rhs_type,
                                      sol_type, svd_l2_solver_tag>(problem);
            delete _precond_R;
            std::cout << "FAILED to create!" << std::endl;
            _precond_R = nullptr;
        }
#else
        //TODO: how to handle?
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Requires FFT support!"));
#endif
    }

    ~accelerated_regression_solver_t() {
        if (_precond_R != nullptr)
            delete _precond_R;
        if (_alt_solver != nullptr)
            delete _alt_solver;
    }

    int solve(const rhs_type& b, sol_type& x) {
        if (_precond_R != nullptr)
            return LSQR(_A, b, x, algorithms::krylov_iter_params_t(),
                *_precond_R);
        else {
            _alt_solver->solve(b, x);
            return 0;
        }
    }
};

/**
 * Specialization: Blendenpik, local input, local output
 */
template <typename ValueType, typename PrecondTag>
class accelerated_regression_solver_t<
    regression_problem_t<El::Matrix<ValueType>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::Matrix<ValueType>,
    El::Matrix<ValueType>,
    blendenpik_tag<PrecondTag> > {

public:

    typedef ValueType value_type;

    typedef El::Matrix<ValueType> matrix_type;
    typedef El::Matrix<ValueType> rhs_type;
    typedef El::Matrix<ValueType> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef El::Matrix<ValueType> precond_type;
    typedef precond_type sketch_type;
    // The assumption is that the sketch is not much bigger than the
    // preconditioner, so we should use the same matrix distribution.

    const int _m;
    const int _n;
    const matrix_type &_A;
    precond_type _R;
    algorithms::inplace_precond_t<sol_type> *_precond_R;

    regression_solver_t<problem_type, rhs_type, sol_type, svd_l2_solver_tag>
    *_alt_solver;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     * @param context Skylark context.
     */
    accelerated_regression_solver_t(const problem_type& problem,
            base::context_t& context) :
        _m(problem.m), _n(problem.n), _A(problem.input_matrix),
        _R(_n, _n) {
        // TODO n < m ???

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF
        int t = 4 * _n;    // TODO parameter.
        double scale = std::sqrt((double)_m / (double)t);

        El::Matrix<ValueType> Ar;
        sketch_type SA(t, _n);
        boost::random::uniform_int_distribution<int> distribution(0, _m- 1);

        Ar = _A;
        double condest = 0;
        int attempts = 0;
        do {
            // TODO parameter of how many rounds
            sketch::RFUT_t<El::Matrix<value_type>,
                           typename sketch::fft_futs<value_type>::DCT_t,
                           utility::rademacher_distribution_t<double> >
                F(_m, context);
            F.apply(Ar, Ar, sketch::columnwise_tag());

            // TODO use sampling matrix!
            std::vector<int> samples =
                context.generate_random_samples_array(t, distribution);
            for (int j = 0; j < Ar.Width(); j++)
                for (int i = 0; i < t; i++) {
                    int row = samples[i];
                    SA.Set(i, j, scale * Ar.Get(row, j));
                }

            condest = flinl2_internal::build_precond(SA, _R, _precond_R,
                PrecondTag());
            attempts++;
        } while (condest > 1e14 && attempts < 3); // TODO parameters

        if (condest <= 1e14)
            _alt_solver = nullptr;
        else {
            _alt_solver =
                new regression_solver_t<problem_type,
                                      rhs_type,
                                      sol_type, svd_l2_solver_tag>(problem);
            delete _precond_R;
            std::cout << "FAILED to create!" << std::endl;
            _precond_R = nullptr;
        }
#else
        //TODO: how to handle?
        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "Requires FFT support!"));
#endif
    }

    ~accelerated_regression_solver_t() {
        if (_precond_R != nullptr)
            delete _precond_R;
        if (_alt_solver != nullptr)
            delete _alt_solver;
    }

    int solve(const rhs_type& b, sol_type& x) {
        if (_precond_R != nullptr)
            return LSQR(_A, b, x, algorithms::krylov_iter_params_t(),
                *_precond_R);
        else {
            _alt_solver->solve(b, x);
            return 0;
        }
    }
};


/**
 * Specialization: LSRN, [VC/VR,STAR] input, [STAR, STAR] solution.
 */
template <typename ValueType, El::Distribution VD,
          typename PrecondTag>
class accelerated_regression_solver_t<
    regression_problem_t<El::DistMatrix<ValueType, VD, El::STAR>,
                         linear_tag, l2_tag, no_reg_tag>,
    El::DistMatrix<ValueType, VD, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    lsrn_tag<PrecondTag> > {

public:

    typedef ValueType value_type;

    typedef El::DistMatrix<ValueType, VD, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, VD, El::STAR> rhs_type;
    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> sol_type;

    typedef regression_problem_t<matrix_type,
                                 linear_tag, l2_tag, no_reg_tag> problem_type;

private:

    typedef El::DistMatrix<ValueType, El::STAR, El::STAR> precond_type;
    typedef precond_type sketch_type;
    // The assumption is that the sketch is not much bigger than the
    // preconditioner, so we should use the same matrix distribution.

    const int _m;
    const int _n;
    const matrix_type &_A;
    bool _use_lsqr;
    double _sigma_U, _sigma_L;
    precond_type _R;
    algorithms::inplace_precond_t<sol_type> *_precond_R;
    algorithms::krylov_iter_params_t _params;

public:
    /**
     * Prepares the regressor to quickly solve given a right-hand side.
     *
     * @param problem Problem to solve given right-hand side.
     * @param context Skylark context.
     */
    accelerated_regression_solver_t(const problem_type& problem, base::context_t& context) :
        _m(problem.m), _n(problem.n), _A(problem.input_matrix),
        _R(_n, _n, problem.input_matrix.Grid()) {
        // TODO n < m ???

        int t = 4 * _n;    // TODO parameter.
        double epsilon = 1e-14;  // TODO parameter
        double delta = 1e-6; // TODO parameter

        sketch::JLT_t<matrix_type, sketch_type> S(_m, t, context);
        sketch_type SA(t, _n);
        S.apply(_A, SA, sketch::columnwise_tag());
        flinl2_internal::build_precond(SA, _R, _precond_R, PrecondTag());

        // Select alpha so that probability of failure is delta.
        // If alpha is too big, we need to use LSQR (although ill-conditioning
        // might be very severe so to prevent convergence).
        double alpha = std::sqrt(2 * std::log(2.0 / delta) / t);
        if (alpha >= (1 - std::sqrt(_n / t)))
            _use_lsqr = true;
        else {
            _use_lsqr = false;
            _sigma_U = std::sqrt(t) / ((1 - alpha) * std::sqrt(t)
                - std::sqrt(_n));
            _sigma_L = std::sqrt(t) / ((1 + alpha) * std::sqrt(t)
                + std::sqrt(_n));
        }
    }

    ~accelerated_regression_solver_t() {
        delete _precond_R;
    }

    int solve(const rhs_type& b, sol_type& x) {
        int ret;
        if (_use_lsqr)
            ret = LSQR(_A, b, x, _params, *_precond_R);
        else {
            ChebyshevLS(_A, b, x,  _sigma_L, _sigma_U,
                _params, *_precond_R);
            ret = -6; // TODO! - check!
        }
        return ret; // TODO!
    }
};



} } /** namespace skylark::algorithms */

#endif // SKYLARK_ACCELERATED_LINEARL2_REGRESSION_SOLVER_ELEMENTAL_HPP
