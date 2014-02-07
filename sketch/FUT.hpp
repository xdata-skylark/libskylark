#ifndef SKYLARK_FUT_HPP
#define SKYLARK_FUT_HPP

#include "../config.h"

//TODO: currently also needs elemental
#if SKYLARK_HAVE_FFTW

#include <elemental.hpp>
#include <fftw3.h>

#include "context.hpp"
#include "transforms.hpp"

namespace skylark { namespace sketch {

template < typename ValueType,
           typename PlanType, typename KindType,
           KindType Kind, KindType KindInverse,
           PlanType (*PlanFun)(int, ValueType*, ValueType*, KindType, unsigned),
           void (*ExecuteFun)(PlanType, ValueType*, ValueType*),
           void (*DestroyFun)(PlanType),
           int ScaleVal >
class fftw_r2r_fut_t {

    void apply_impl(elem::Matrix<ValueType>& A,
                    skylark::sketch::columnwise_tag) const {
        // The efficiency of the following depends if FFTW wisdom has been
        // built in advance. It is also purely sequential (may want to do
        // something different in the final version).
        // SO: this is a PRELIMINARY version.

        double* AA = A.Buffer();
        PlanType plan = PlanFun(A.Height(), AA, AA, Kind,
                                FFTW_UNALIGNED | FFTW_ESTIMATE);

        // TODO: detect failure to form plan.
        for (int j = 0; j < A.Width(); j++)
            ExecuteFun(plan, AA + j * A.LDim(), AA + j * A.LDim());
        DestroyFun(plan);
    }

    void apply_inverse_impl(elem::Matrix<ValueType>& A,
                            skylark::sketch::columnwise_tag) const {

        // The efficiency of the following depends if FFTW wisdom has been
        // built in advance. It is also purely sequential (may want to do
        // something different in the final version).
        // SO: this is a PRELIMINARY version.

        double* AA = A.Buffer();
        PlanType plan = PlanFun(A.Height(), AA, AA, KindInverse,
                                FFTW_UNALIGNED | FFTW_ESTIMATE);

        // TODO: detect failure to form plan.
        for (int j = 0; j < A.Width(); j++)
            ExecuteFun(plan, AA + j * A.LDim(), AA + j * A.LDim());
        DestroyFun(plan);
    }

    void apply_impl(elem::Matrix<ValueType>& A,
                    skylark::sketch::rowwise_tag) const {

        // The efficiency of the following depends if FFTW wisdom has been
        // built in advance. It is also purely sequential (may want to do
        // something different in the final version).
        // SO: this is a PRELIMINARY version.

        // Using transpositions instead of moving
        // to the advanced interface of FFTW
        elem::Matrix<ValueType> matrix(A.Grid());
        elem::Transpose(A, matrix);
        double* matrix_buffer = matrix.Buffer();
        PlanType plan = PlanFun(matrix.Height(), matrix_buffer, matrix_buffer,
            Kind, FFTW_UNALIGNED | FFTW_ESTIMATE);

        // TODO: detect failure to form plan.
        for (int j = 0; j < matrix.Width(); j++)
            ExecuteFun(plan, matrix_buffer + j * matrix.LDim(),
                matrix_buffer + j * matrix.LDim());
        DestroyFun(plan);
        elem::Transpose(matrix, A);
    }

    void apply_inverse_impl(elem::Matrix<ValueType>& A,
                            skylark::sketch::rowwise_tag) const {
        // The efficiency of the following depends if FFTW wisdom has been
        // built in advance. It is also purely sequential (may want to do
        // something different in the final version).
        // SO: this is a PRELIMINARY version.

        // Using transpositions instead of moving
        // to the advanced interface of FFTW
        elem::Matrix<ValueType> matrix(A.Grid());
        elem::Transpose(A, matrix);
        double* matrix_buffer = matrix.Buffer();
        PlanType plan = PlanFun(matrix.Height(), matrix_buffer, matrix_buffer,
            KindInverse, FFTW_UNALIGNED | FFTW_ESTIMATE);

        // TODO: detect failure to form plan.
        for (int j = 0; j < matrix.Width(); j++)
            ExecuteFun(plan, matrix_buffer + j * matrix.LDim(),
                matrix_buffer + j * matrix.LDim());
        DestroyFun(plan);
        elem::Transpose(matrix, A);
    }

    double scale_impl(const elem::Matrix<ValueType>& A,
                      skylark::sketch::columnwise_tag) const {
        return 1 / sqrt((double)ScaleVal * A.Height());
    }

    double scale(const elem::Matrix<ValueType>& A,
                 skylark::sketch::rowwise_tag) const {
        return 1 / sqrt((double)ScaleVal * A.Width());
    }

public:

    template <typename Dimension>
    void apply(elem::Matrix<ValueType>& A, Dimension dimension) const {
        return apply_impl (A, dimension);
    }

    template <typename Dimension>
    void apply_inverse(elem::Matrix<ValueType>& A, Dimension dimension) const {
        return apply_inverse_impl (A, dimension);
    }

    template <typename Dimension>
    double scale(const elem::Matrix<ValueType>& A, Dimension dimension) const {
        return scale_impl(A, dimension);
    }
};

template<typename ValueType>
struct fft_futs {

};

template<>
struct fft_futs<double> {
    typedef fftw_r2r_fut_t <
            double, fftw_plan, fftw_r2r_kind, FFTW_REDFT10, FFTW_REDFT01,
            fftw_plan_r2r_1d, fftw_execute_r2r, fftw_destroy_plan, 2 > DCT;

    typedef fftw_r2r_fut_t <
            double, fftw_plan, fftw_r2r_kind, FFTW_DHT, FFTW_DHT,
            fftw_plan_r2r_1d, fftw_execute_r2r, fftw_destroy_plan, 1 > DHT;
};

template<>
struct fft_futs<float> {
    typedef fftw_r2r_fut_t <
            float, fftwf_plan, fftwf_r2r_kind, FFTW_REDFT10, FFTW_REDFT01,
            fftwf_plan_r2r_1d, fftwf_execute_r2r, fftwf_destroy_plan, 2 > DCT;

    typedef fftw_r2r_fut_t <
            float, fftwf_plan, fftwf_r2r_kind, FFTW_DHT, FFTW_DHT,
            fftwf_plan_r2r_1d, fftwf_execute_r2r, fftwf_destroy_plan, 1 > DHT;
};


} } /** namespace skylark::sketch */

#endif // SKYLARK_HAVE_FFTW

#endif // SKYLARK_FUT_HPP
