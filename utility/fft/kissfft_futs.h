
#include "kissfft.hh"

template < typename ValueType,
           int ScaleVal >
struct kissfft_r2r_fut_t {

    kissfft_r2r_fut_t< ValueType, ScaleVal > (int N) : _N(N) {
        bool is_inverse_fft = true;
        fft = new kissfft<ValueType>  (_N*4, true);
        ifft = new kissfft<ValueType> (_N*4, false);

        fft_in = new std::complex<ValueType> [_N*4];
        fft_out = new std::complex<ValueType> [_N*4];
    
    }
    
    virtual ~kissfft_r2r_fut_t < ValueType, ScaleVal > () {
        free(fft);
        free(ifft);

        free(fft_in);
        free(fft_out);
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

    void expand4(ValueType* values) const {
        
        for (int i = 0; i < _N*4; ++i) {
            fft_in[i] = std::complex<ValueType>(0,0);
        }

        for (int i = 0; i < _N; ++i) {
            int pos = 2*i+1;
            std::complex<ValueType> cplx(values[i],0);
            fft_in[pos] = cplx;
            fft_in[_N*4-pos] = cplx;
        }
    }

    void reduce4(ValueType* values) const {
        
        for(int i = 0; i <_N; ++i)
            values[i] = fft_out[i].real();
            
    }

    void iexpand4(ValueType* values) const {
        // from [A, B, C, D] 
        // to [A, B, C, D, 0, -D, -C, -B, -A, -B, -C, -D, 0, D, C, B]

        int pos = 0;
        for (int i = 0; i < _N; ++i) {
            fft_in[pos] = std::complex<ValueType>(values[i],0);
            ++pos;
        }

        fft_in[pos] = std::complex<ValueType>(0,0);
        ++pos;

        for(int i = _N-1; i >= 0; --i) {
            fft_in[pos] = std::complex<ValueType>(-values[i],0);
            ++pos;
        }

        for(int i = 1; i < _N; ++i) {
            fft_in[pos] = std::complex<ValueType>(-values[i],0);
            ++pos;
        }

        fft_in[pos] = std::complex<ValueType>(0,0);
        ++pos;

        for(int i = _N-1; i >= 0; --i) {
            fft_in[pos] = std::complex<ValueType>(values[i],0);
            ++pos;
        }
    }

    // DCT Implementation
    // http://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::columnwise_tag) const {
        
        ValueType* AA = A.Buffer();

        for(int i = 0; i < A.Width(); ++i) {
            expand4((AA + i * A.LDim()));
            
            fft->transform(fft_in, fft_out);

            reduce4((AA + i * A.LDim()));
        }
    }
    

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::columnwise_tag) const {
        // TODO(Jordi): Implement DCT III
    }

    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::rowwise_tag) const {

        El::Matrix<ValueType> matrix;
        El::Transpose(A, matrix);  
        
        ValueType* matrixBuffer = matrix.Buffer();

        for(int i = 0; i < matrix.Width(); ++i) {
            expand4((matrixBuffer + i * matrix.LDim()));
            fft->transform(fft_in, fft_out);
            reduce4((matrixBuffer + i * matrix.LDim()));
        }
        El::Transpose(matrix, A);

    }

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::rowwise_tag) const {
        // TODO(Jordi): Implement DCT III
    }

    const int _N;
    
    kissfft<ValueType> *fft;
    kissfft<ValueType> *ifft;

    std::complex<ValueType>* fft_in;
    std::complex<ValueType>* fft_out;

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