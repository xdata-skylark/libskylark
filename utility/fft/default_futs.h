

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


template<>
struct fft_futs<double> {

  struct empty_t {

    empty_t(int N) { 
      SKYLARK_THROW_EXCEPTION (
	base::sketch_exception()
              << base::error_msg(
                 "Double precision fftw has not been compiled."));
    }

    template <typename Dimension>
    void apply(El::Matrix<double>& A, Dimension dimension) const { }

    template <typename Dimension>
    void apply_inverse(El::Matrix<double>& A, Dimension dimension) const { }

    double scale() const { return 0.0; }

  };

  typedef empty_t DCT_t;
  typedef empty_t DHT_t;
};