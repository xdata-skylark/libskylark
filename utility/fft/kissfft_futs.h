
#include "kissfft.hh"

template < typename ValueType,
           int ScaleVal >
struct kissfft_r2r_fut_t {

    kissfft_r2r_fut_t< ValueType, ScaleVal > (int N) : _N(N) {

        bool is_inverse_fft = true;
        fft = new kissfft<ValueType>  (N, !is_inverse_fft);
        //ifft = new  kissfft<std::complex<ValueType> >  (N, is_inverse_fft);

        //fft_in.resize();  // = new std::complex<ValueType> [N*4];
        //fft_out.resize(N*4); // = new std::complex<ValueType> [N*4];
    }
    
    virtual ~kissfft_r2r_fut_t < ValueType, ScaleVal > () {
        //free(fft);
        //free(ifft);
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

    void fillInVector(ValueType* values, 
        std::vector<std::complex<ValueType> > &fft_in) const {
        // values => [a, b, c, d]
        // fft_in [0, a, 0, b, 0,  c,  0,  d,  0,  d,  0,  c, 0, b, 0, a]
        for (int i = 0; i < 4*_N; ++i) {
            ValueType realPart = 0;
            if (i & 1) {
                if (i < _N) {
                    realPart = (ValueType) values[i/2];
                } else {
                    realPart = (ValueType) values[_N - (i/2-_N)];
                }
            }
            fft_in[i] = std::complex<ValueType> (realPart, 0.0);
        }

    }

    // DCT Implementation
    // http://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::columnwise_tag) const {
        
        std::vector<std::complex<ValueType> > fft_in (_N*4);
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++) {
            fillInVector((AA + j * A.LDim()), fft_in);
            // [A, B, C, D, 0, -D, -C, -B, -A, -B, -C, -D, 0, D, C, B]
            fft->transform(&fft_in[0], &fft_in[0]);

            // real part fo [A, B, C, D]
            for (int i = 0; i < _N; ++i) 
                *(AA + j * A.LDim() + i) = fft_in[i].real();

        }
    }
    

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::columnwise_tag) const {

        std::vector<std::complex<ValueType> > fft_in (_N*4);
                ValueType* AA = A.Buffer();
                int j;

        #       ifdef SKYLARK_HAVE_OPENMP
        #       pragma omp parallel for private(j)
        #       endif
                for (j = 0; j < A.Width(); j++) {
                    fillInVector((AA + j * A.LDim()), fft_in);
                    // [A, B, C, D, 0, -D, -C, -B, -A, -B, -C, -D, 0, D, C, B]
                    ifft->transform(&fft_in[0], &fft_in[0]);

                    // real part fo [A, B, C, D]
                    for (int i = 0; i < _N; ++i) 
                        *(AA + j * A.LDim() + i) = fft_in[i].real();
        }
    }

    void apply_impl(El::Matrix<ValueType>& A,
                    skylark::sketch::rowwise_tag) const {

       std::vector<std::complex<ValueType> > fft_in (_N*4);
        El::Matrix<ValueType> matrix;
        El::Transpose(A, matrix);
        ValueType* matrix_buffer = matrix.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < matrix.Width(); j++) {
            fillInVector((matrix_buffer + j * matrix.LDim()), fft_in);
            // [A, B, C, D, 0, -D, -C, -B, -A, -B, -C, -D, 0, D, C, B]
            fft->transform(&fft_in[0], &fft_in[0]);

            // real part fo [A, B, C, D]
            for (int i = 0; i < _N; ++i) 
                *(matrix_buffer + j * matrix.LDim() + i) = fft_in[i].real();
        }
        El::Transpose(matrix, A);
    }

    void apply_inverse_impl(El::Matrix<ValueType>& A,
                            skylark::sketch::rowwise_tag) const {
        
        std::vector<std::complex<ValueType> > fft_in (_N*4);
        El::Matrix<ValueType> matrix;
        El::Transpose(A, matrix);
        ValueType* matrix_buffer = matrix.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < matrix.Width(); j++) {
            fillInVector((matrix_buffer + j * matrix.LDim()), fft_in);
            // [A, B, C, D, 0, -D, -C, -B, -A, -B, -C, -D, 0, D, C, B]
            ifft->transform(&fft_in[0], &fft_in[0]);

            // real part fo [A, B, C, D]
            for (int i = 0; i < _N; ++i) 
                *(matrix_buffer + j * matrix.LDim() + i) = fft_in[i].real();
        }
        El::Transpose(matrix, A);
   
    }

    const int _N;
    kissfft<ValueType> *fft;
    kissfft<ValueType> *ifft;

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