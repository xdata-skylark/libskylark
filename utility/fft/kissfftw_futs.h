
#include "kiss_fft.h"
#include "kiss_fftr.h"


template < typename ValueType,
           int ScaleVal >
struct kissfft_r2r_fut_t {

    kissfft_r2r_fut_t< ValueType, ScaleVal > (int N) : _N(N) {

        bool is_inverse_fft = true;

        _cfg = kiss_fftr_alloc(N, is_inverse_fft, 0, 0);
        _inverse_cfg = kiss_fftr_alloc(N, (!is_inverse_fft), 0, 0);

    }
    
    virtual ~kissfft_r2r_fut_t < ValueType, ScaleVal > () {
        free(_cfg);
        free(_inverse_cfg);
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
        double fft_in[_N*4];
        kiss_fft_cpx fft_out[_N*4];
        int j;

        for (int i = 0; i < 4*_N; ++i) {
            if (i & 1) {
                if (i < _N) {
                    fft_in[i] = AA[i/2]*A.LDim();
                } else {
                    fft_in[i] = AA[_N - (i/2-_N)]*A.LDim();
                }
            } else {
                fft_in[i] = 0;
            }
        }

        for (int i = 0; i < _N*4; ++i) {
            std::cout << fft_in[i] << ", ";
        }
        std::cout << std::endl;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++){
            kiss_fftr(_cfg,  fft_in, fft_out);
        }


        //kiss_fftr(_cfg,  AA + j * A.LDim(), AA + j * A.LDim());
            // fft_out
    }

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::columnwise_tag) const {

    }

    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::rowwise_tag) const {
       
    }

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::rowwise_tag) const {
   
    }

    const int _N;
    kiss_fftr_cfg _cfg, _inverse_cfg; 
};


template<>
struct fft_futs<double> {
    typedef kissfft_r2r_fut_t <double, 2> DCT_t;

    typedef kissfft_r2r_fut_t <double, 1> DHT_t;
};

template<>
struct fft_futs<float> {
    typedef kissfft_r2r_fut_t <float, 2> DCT_t;

    typedef kissfft_r2r_fut_t <float, 1> DHT_t;
};