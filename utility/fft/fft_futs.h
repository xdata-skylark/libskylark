#ifndef SKYLARK_FFT_FUTS_HPP
#define SKYLARK_FFT_FUTS_HPP


#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF
    #include "fftw_futs.h"
#elif SKYLARK_HAVE_KISSFFT
    #include "kissfft_futs.h"
#else
    #include "default_futs.h"
#endif // fft_futs



#endif /** SKYLARK_FFT_FUTS_HPP */
