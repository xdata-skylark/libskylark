#ifndef SKYLARK_PPT_ELEMENTAL_HPP
#define SKYLARK_PPT_ELEMENTAL_HPP

#include <elemental.hpp>
#include <fftw3.h>

#include "context.hpp"
#include "transforms.hpp"
#include "PPT_data.hpp"
#include "../utility/exception.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization for local to local.
 */
template<> 
struct PPT_t <
    elem::Matrix<double>,
    elem::Matrix<double> > :
        public PPT_data_t<double>,
        virtual public sketch_transform_t<elem::Matrix<double>,
                                          elem::Matrix<double> >{

    typedef double value_type;
    typedef elem::Matrix<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef PPT_data_t<double> base_data_t;

    /**
     * Regular constructor
     */
    PPT_t(int N, int S, int q, double c, double gamma,
        skylark::sketch::context_t& context)
        : base_data_t (N, S, q, c, gamma, context)  {

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

    ~PPT_t() {
        fftw_destroy_plan(_fftw_fplan);
        fftw_destroy_plan(_fftw_bplan);
    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    PPT_t(const PPT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : base_data_t(other) {

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

    /**
     * Constructor from data
     */
    PPT_t(const base_data_t& other_data)
        : base_data_t(other_data) {

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
        matrix_type Av(N, 1);
        matrix_type SAv(S, 1);

        std::complex<double> *FW = new std::complex<double>[S];
        std::complex<double> *P = new std::complex<double>[S];

        for(int i = 0; i < A.Width(); i++) {
            elem::LockedView(Av, A, 0, i, A.Height(), 1);
            elem::View(SAv, sketch_of_A, 0, i, A.Height(), 1);

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
                fftw_execute_dft_r2c(_fftw_fplan, W.Buffer(),
                    reinterpret_cast<fftw_complex*>(FW));
                for(int j = 0; j < S; j++)
                    P[j] *= FW[j];
            }

            // In FFTW, both fft and ifft are not scaled.
            // That is norm(ifft(fft(x)) = norm(x) * #els(x).
            for(int j = 0; j < S; j++)
                P[j] /= (double)S;

            fftw_execute_dft_c2r(_fftw_bplan,
                reinterpret_cast<fftw_complex*>(P), SAv.Buffer());
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

        // TODO
    }

    int get_N() const { return base_data_t::_N; } /**< Get input dimesion. */
    int get_S() const { return base_data_t::_S; } /**< Get output dimesion. */


protected:

    typedef typename base_data_t::_CWT_data_t _CWT_data_t;
    typedef CWT_t<matrix_type, output_matrix_type> _CWT_t;

    fftw_plan _fftw_fplan, _fftw_bplan;
    std::list<_CWT_t> _cwts;

};

} } /** namespace skylark::sketch */

#endif // PPT_ELEMENTAL_HPP
