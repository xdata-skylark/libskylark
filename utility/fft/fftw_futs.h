
#include <fftw3.h>
template < typename ValueType,
           typename PlanType, typename KindType,
           KindType Kind, KindType KindInverse,
           PlanType (*PlanFun)(int, ValueType*, ValueType*, KindType, unsigned),
           void (*ExecuteFun)(PlanType, ValueType*, ValueType*),
           void (*DestroyFun)(PlanType),
           int ScaleVal >
struct fftw_r2r_fut_t {

    fftw_r2r_fut_t<ValueType,
                   PlanType, KindType,
                   Kind, KindInverse,
                   PlanFun, ExecuteFun, DestroyFun, ScaleVal>(int N) : _N(N) {
        ValueType *tmp = new ValueType[N];
        _plan = PlanFun(N, tmp, tmp, Kind, FFTW_UNALIGNED | FFTW_ESTIMATE);
        _plan_inverse = PlanFun(N, tmp, tmp, KindInverse,
            FFTW_UNALIGNED | FFTW_ESTIMATE);
        delete[] tmp;

        // TODO: detect failure to form plans.
    }

    virtual ~fftw_r2r_fut_t<ValueType,
                            PlanType, KindType,
                            Kind, KindInverse,
                            PlanFun, ExecuteFun, DestroyFun, ScaleVal>() {
        DestroyFun(_plan);
        DestroyFun(_plan_inverse);
    }


    template <typename Dimension>
    void apply(El::Matrix<ValueType>& A, Dimension dimension) const {
        return apply_impl (A, dimension);
    }

    template <typename Dimension>
    void apply_inverse(El::Matrix<ValueType>& A, Dimension dimension) const {
        return apply_inverse_impl (A, dimension);
    }

    double scale() const {
        return 1 / sqrt((double)ScaleVal * _N);
    }

private:

    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::columnwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            ExecuteFun(_plan, AA + j * A.LDim(), AA + j * A.LDim());
    }

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::columnwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            ExecuteFun(_plan_inverse, AA + j * A.LDim(), AA + j * A.LDim());
    }

    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::rowwise_tag) const {
        // Using transpositions instead of moving to the advanced interface
        // of FFTW
        El::Matrix<ValueType> matrix;
        El::Transpose(A, matrix);
        ValueType* matrix_buffer = matrix.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < matrix.Width(); j++)
            ExecuteFun(_plan, matrix_buffer + j * matrix.LDim(),
                matrix_buffer + j * matrix.LDim());
        El::Transpose(matrix, A);
    }

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::rowwise_tag) const {
        // Using transpositions instead of moving to the advanced interface
        // of FFTW
        El::Matrix<ValueType> matrix;
        El::Transpose(A, matrix);
        ValueType* matrix_buffer = matrix.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < matrix.Width(); j++)
            ExecuteFun(_plan_inverse, matrix_buffer + j * matrix.LDim(),
                matrix_buffer + j * matrix.LDim());
        El::Transpose(matrix, A);
    }

private:
    const int _N;
    PlanType _plan, _plan_inverse;
};


#if SKYLARK_HAVE_FFTW
template<>
struct fft_futs<double> {
    typedef fftw_r2r_fut_t <
            double, fftw_plan, fftw_r2r_kind, FFTW_REDFT10, FFTW_REDFT01,
            fftw_plan_r2r_1d, fftw_execute_r2r, fftw_destroy_plan, 2 > DCT_t;

    typedef fftw_r2r_fut_t <
            double, fftw_plan, fftw_r2r_kind, FFTW_DHT, FFTW_DHT,
            fftw_plan_r2r_1d, fftw_execute_r2r, fftw_destroy_plan, 1 > DHT_t;
};

#endif

#if SKYLARK_HAVE_FFTWF

template<>
struct fft_futs<float> {
    typedef fftw_r2r_fut_t <
            float, fftwf_plan, fftwf_r2r_kind, FFTW_REDFT10, FFTW_REDFT01,
            fftwf_plan_r2r_1d, fftwf_execute_r2r, fftwf_destroy_plan, 2 > DCT_t;

    typedef fftw_r2r_fut_t <
            float, fftwf_plan, fftwf_r2r_kind, FFTW_DHT, FFTW_DHT,
            fftwf_plan_r2r_1d, fftwf_execute_r2r, fftwf_destroy_plan, 1 > DHT_t;
};
#else
template<>
struct fft_futs<float> {

  struct empty_t {

    empty_t(int N) { 

      SKYLARK_THROW_EXCEPTION (
	base::sketch_exception()
              << base::error_msg(
                 "Single precision fftw has not been compiled."));
    }

    template <typename Dimension>
    void apply(El::Matrix<float>& A, Dimension dimension) const { }

    template <typename Dimension>
    void apply_inverse(El::Matrix<float>& A, Dimension dimension) const { }

    double scale() const { return 0.0; }

  };

  typedef empty_t DCT_t;
  typedef empty_t DHT_t;
};
#endif

