#ifndef SKYLARK_PPT_ELEMENTAL_HPP
#define SKYLARK_PPT_ELEMENTAL_HPP

#include "../config.h"

#if (SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF) && SKYLARK_HAVE_ELEMENTAL

#include <elemental.hpp>
#include <fftw3.h>

#include "context.hpp"
#include "transforms.hpp"
#include "PPT_data.hpp"
#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * Some hacking to make use of fftw more generic.
 */
namespace internal {

template <typename T>
struct fftw {

};

#ifdef SKYLARK_HAVE_FFTW

template <>
struct fftw<double> {
    typedef fftw_complex complex_t;
    typedef fftw_plan plan_t;

    typedef plan_t (*fplanfun_t)(int, double*, complex_t*, unsigned);
    static fplanfun_t fplanfun;
    typedef plan_t (*bplanfun_t)(int, complex_t*, double*, unsigned);
    static bplanfun_t bplanfun;
    typedef void (*destroyfun_t)(plan_t);
    static destroyfun_t destroyfun;
    typedef void (*executeffun_t)(plan_t, double*, complex_t*);
    static executeffun_t executeffun;
    typedef void (*executebfun_t)(plan_t, complex_t*, double*);
    static executebfun_t executebfun;
};

fftw<double>::fplanfun_t fftw<double>::fplanfun = fftw_plan_dft_r2c_1d;
fftw<double>::bplanfun_t fftw<double>::bplanfun = fftw_plan_dft_c2r_1d;
fftw<double>::destroyfun_t fftw<double>::destroyfun = fftw_destroy_plan;
fftw<double>::executeffun_t fftw<double>::executeffun = fftw_execute_dft_r2c;
fftw<double>::executebfun_t fftw<double>::executebfun = fftw_execute_dft_c2r;

#endif /* SKYLARK_HAVE_FFTW */

#ifdef SKYLARK_HAVE_FFTWF

template <>
struct fftw<float> {
    typedef fftwf_complex complex_t;
    typedef fftwf_plan plan_t;

    typedef plan_t (*fplanfun_t)(int, float*, complex_t*, unsigned);
    static fplanfun_t fplanfun;
    typedef plan_t (*bplanfun_t)(int, complex_t*, float*, unsigned);
    static bplanfun_t bplanfun;
    typedef void (*destroyfun_t)(plan_t);
    static destroyfun_t destroyfun;
    typedef void (*executeffun_t)(plan_t, float*, complex_t*);
    static executffun_t executeffun;
    typedef void (*executebfun_t)(plan_t, complex_t*, float*);
    static executbfun_t executebfun;
};

fftw<float>::fplanfun_t fftw<float>::fplanfun = fftwf_plan_dft_r2c_1d;
fftw<float>::bplanfun_t fftw<float>::bplanfun = fftwf_plan_dft_c2r_1d;
fftw<float>::destroyfun_t fftw<float>::destroyfun = fftwf_destroy_plan;
fftw<float>::executeffun_t fftw<float>::executeffun = fftwf_execute_dft_r2c;
fftw<float>::executebfun_t fftw<float>::executebfun = fftwf_execute_dft_c2r;

#endif /* SKYLARK_HAVE_FFTWF */

}  /** namespace skylark::sketch::internal */

/**
 * Specialization for local to local.
 */
template<typename ValueType> 
struct PPT_t <
    elem::Matrix<ValueType>,
    elem::Matrix<ValueType> > :
        public PPT_data_t<ValueType>,
        virtual public sketch_transform_t<elem::Matrix<ValueType>,
                                          elem::Matrix<ValueType> >{

    typedef ValueType value_type;
    typedef elem::Matrix<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef PPT_data_t<ValueType> base_data_t;

    /**
     * Regular constructor
     */
    PPT_t(int N, int S, int q, double c, double gamma,
        skylark::sketch::context_t& context)
        : base_data_t (N, S, q, c, gamma, context)  {

        build_internal();
    }

    ~PPT_t() {
        internal::fftw<value_type>::destroyfun(_fftw_fplan);
        internal::fftw<value_type>::destroyfun(_fftw_bplan);
    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    PPT_t(const PPT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : base_data_t(other) {

        build_internal();
    }

    /**
     * Constructor from data
     */
    PPT_t(const base_data_t& other_data)
        : base_data_t(other_data) {

        build_internal();
    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {

        // TODO verify sizes etc.
        int S = base_data_t::_S;
        int N = base_data_t::_N;

        matrix_type W(S, 1);
        matrix_type SAv;
        matrix_type Av;

        std::complex<value_type> *FW = new std::complex<value_type>[S];
        std::complex<value_type> *P = new std::complex<value_type>[S];

        // TODO OpenMP parallelization
        for(int i = 0; i < A.Width(); i++) {
            elem::LockedView(Av, A, 0, i, A.Height(), 1);

            for(int j = 0; j < S; j++)
                P[j] = 1.0;

            typename std::list<_CWT_t>::const_iterator it;
            int qc = 0;
            for(it = _cwts.begin(); it != _cwts.end(); it++, qc++) {
                const _CWT_t &C = *it;
                C.apply(Av, W, columnwise_tag());
                elem::Scal(std::sqrt(base_data_t::_gamma), W);
                W.Update(base_data_t::_hash_idx[qc], 0,
                    std::sqrt(base_data_t::_c) * base_data_t::_hash_val[qc]);
                internal::fftw<value_type>::executeffun(_fftw_fplan, W.Buffer(),
                    reinterpret_cast<_fftw_complex_t*>(FW));
                for(int j = 0; j < S; j++)
                    P[j] *= FW[j];
            }

            // In FFTW, both fft and ifft are not scaled.
            // That is norm(ifft(fft(x)) = norm(x) * #els(x).
            for(int j = 0; j < S; j++)
                P[j] /= (value_type)S;

            elem::View(SAv, sketch_of_A, 0, i, A.Height(), 1);
            internal::fftw<value_type>::executebfun(_fftw_bplan,
                reinterpret_cast<_fftw_complex_t*>(P), SAv.Buffer());
        }

        delete FW;
        delete P;
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        // TODO verify sizes etc.
        int S = base_data_t::_S;
        int N = base_data_t::_N;

        matrix_type W(S, 1);
        matrix_type ASv, SATv(S, 1);

        matrix_type Av, ATv;

        std::complex<value_type> *FW = new std::complex<value_type>[S];
        std::complex<value_type> *P = new std::complex<value_type>[S];

        // TODO OpenMP parallelization
        for(int i = 0; i < A.Height(); i++) {
            elem::LockedView(Av, A, i, 0, 1, A.Width());
            elem::Transpose(Av, ATv);

            for(int j = 0; j < S; j++)
                P[j] = 1.0;

            typename std::list<_CWT_t>::const_iterator it;
            int qc = 0;
            for(it = _cwts.begin(); it != _cwts.end(); it++, qc++) {
                const _CWT_t &C = *it;
                C.apply(ATv, W, columnwise_tag());
                elem::Scal(std::sqrt(base_data_t::_gamma), W);
                W.Update(base_data_t::_hash_idx[qc], 0,
                    std::sqrt(base_data_t::_c) * base_data_t::_hash_val[qc]);
                internal::fftw<value_type>::executeffun(_fftw_fplan, W.Buffer(),
                    reinterpret_cast<_fftw_complex_t*>(FW));
                for(int j = 0; j < S; j++)
                    P[j] *= FW[j];
            }

            // In FFTW, both fft and ifft are not scaled.
            // That is norm(ifft(fft(x)) = norm(x) * #els(x).
            for(int j = 0; j < S; j++)
                P[j] /= (value_type)S;

            internal::fftw<value_type>::executebfun(_fftw_bplan,
                reinterpret_cast<_fftw_complex_t*>(P), SATv.Buffer()); 
            elem::View(ASv, sketch_of_A, i, 0, 1, sketch_of_A.Width());
            elem::Transpose(SATv, ASv);
        }
    }

    int get_N() const { return base_data_t::_N; } /**< Get input dimesion. */
    int get_S() const { return base_data_t::_S; } /**< Get output dimesion. */


protected:

    typedef typename base_data_t::_CWT_data_t _CWT_data_t;
    typedef CWT_t<matrix_type, output_matrix_type> _CWT_t;

    typedef typename internal::fftw<value_type>::complex_t _fftw_complex_t;
    typedef typename internal::fftw<value_type>::plan_t _fftw_plan_t;

    _fftw_plan_t _fftw_fplan, _fftw_bplan;
    std::list<_CWT_t> _cwts;

    void build_internal() {
        int S = base_data_t::_S;

        for(typename std::list<_CWT_data_t>::iterator it =
                base_data_t::_cwts_data.begin();
            it != base_data_t::_cwts_data.end(); it++)
            _cwts.push_back(_CWT_t(*it));

        double *dtmp = new double[S];
        std::complex<double> *ctmp = new std::complex<double>[S];
        _fftw_fplan = fftw_plan_dft_r2c_1d(S, dtmp,
            reinterpret_cast<fftw_complex*>(ctmp),
            FFTW_UNALIGNED | FFTW_ESTIMATE);
        _fftw_bplan = fftw_plan_dft_c2r_1d(S,
            reinterpret_cast<fftw_complex*>(ctmp), dtmp,
            FFTW_UNALIGNED | FFTW_ESTIMATE);
        delete dtmp;
        delete ctmp;
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_HAVE_FFTW && SKYLARK_HAVE_ELEMENTAL

#endif // PPT_ELEMENTAL_HPP
