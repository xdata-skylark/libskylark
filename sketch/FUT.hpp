#ifndef SKYLARK_FUT_HPP
#define SKYLARK_FUT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#if SKYLARK_HAVE_FFTW

#include <fftw3.h>

namespace skylark { namespace sketch {

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
#ifdef SKYLARK_HAVE_OPENMP
#pragma omp parallel for private(j)
#endif
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

template<typename ValueType>
struct fft_futs {

};

template<>
struct fft_futs<double> {
    typedef fftw_r2r_fut_t <
            double, fftw_plan, fftw_r2r_kind, FFTW_REDFT10, FFTW_REDFT01,
            fftw_plan_r2r_1d, fftw_execute_r2r, fftw_destroy_plan, 2 > DCT_t;

    typedef fftw_r2r_fut_t <
            double, fftw_plan, fftw_r2r_kind, FFTW_DHT, FFTW_DHT,
            fftw_plan_r2r_1d, fftw_execute_r2r, fftw_destroy_plan, 1 > DHT_t;
};

template<>
struct fft_futs<float> {
    typedef fftw_r2r_fut_t <
            float, fftwf_plan, fftwf_r2r_kind, FFTW_REDFT10, FFTW_REDFT01,
            fftwf_plan_r2r_1d, fftwf_execute_r2r, fftwf_destroy_plan, 2 > DCT_t;

    typedef fftw_r2r_fut_t <
            float, fftwf_plan, fftwf_r2r_kind, FFTW_DHT, FFTW_DHT,
            fftwf_plan_r2r_1d, fftwf_execute_r2r, fftwf_destroy_plan, 1 > DHT_t;
};


} } /** namespace skylark::sketch */

#endif // SKYLARK_HAVE_FFTW

#if SKYLARK_HAVE_SPIRALWHT


extern "C" {
#include <spiral_wht.h>
}

namespace skylark { namespace sketch {

template<typename ValueType>
struct WHT_t {

};

template<>
struct WHT_t<double> {

    typedef double value_type;

    WHT_t<double>(int N) : _N(N) {
        // TODO check that N is a power of two.
        _tree = wht_get_tree(ceil(log(N) / log(2)));
    }

    ~WHT_t<double>() {
        delete _tree;
    }

    template <typename Dimension>
    void apply(El::Matrix<value_type>& A, Dimension dimension) const {
        return apply_impl (A, dimension);
    }

    template <typename Dimension>
    void apply_inverse(El::Matrix<value_type>& A, Dimension dimension) const {
        return apply_inverse_impl (A, dimension);
    }

    double scale() const {
        return 1 / sqrt(_N);
    }

private:

    void apply_impl(El::Matrix<value_type>& A,
                    skylark::sketch::columnwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, 1, AA + j * A.LDim());
    }

    void apply_inverse_impl(El::Matrix<value_type>& A,
        skylark::sketch::columnwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, 1, AA + j * A.LDim());
    }

    void apply_impl(El::Matrix<value_type>& A,
        skylark::sketch::rowwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, A.Height(), AA + j);
        // Not sure stride is used correctly here.
    }

    void apply_inverse_impl(El::Matrix<value_type>& A,
        skylark::sketch::rowwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, A.Height(), AA + j);
        // Not sure stride is used correctly here.
    }


    const int _N;
    Wht *_tree;
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HAVE_SPIRALWHT

#endif // SKYLARK_FUT_HPP
