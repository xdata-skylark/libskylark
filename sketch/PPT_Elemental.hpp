#ifndef SKYLARK_PPT_ELEMENTAL_HPP
#define SKYLARK_PPT_ELEMENTAL_HPP

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF

#include <fftw3.h>

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
    static executeffun_t executeffun;
    typedef void (*executebfun_t)(plan_t, complex_t*, float*);
    static executebfun_t executebfun;
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
    El::Matrix<ValueType>,
    El::Matrix<ValueType> > :
        public PPT_data_t,
        virtual public sketch_transform_t<El::Matrix<ValueType>,
                                          El::Matrix<ValueType> >{

    typedef ValueType value_type;
    typedef El::Matrix<value_type> matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;

    typedef PPT_data_t data_type;
    typedef data_type::params_t params_t;

    PPT_t(int N, int S, int q, double c, double gamma, base::context_t& context)
        : data_type (N, S, q, c, gamma, context)  {

        build_internal();
    }

    PPT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context)  {

        build_internal();
    }

    PPT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {
        build_internal();
    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    PPT_t(const PPT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

        build_internal();
    }

    PPT_t(const data_type& other_data)
        : data_type(other_data) {

        build_internal();
    }

    ~PPT_t() {
        internal::fftw<value_type>::destroyfun(_fftw_fplan);
        internal::fftw<value_type>::destroyfun(_fftw_bplan);
    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {

        // TODO verify sizes etc.
        const int S = data_type::_S;
        const int N = data_type::_N;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel
#       endif
        {
        matrix_type W(S, 1);
        matrix_type SAv;
        matrix_type Av;

        std::complex<value_type> *FW = new std::complex<value_type>[S];
        std::complex<value_type> *P = new std::complex<value_type>[S];

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp for
#       endif
        for(int i = 0; i < A.Width(); i++) {
            El::LockedView(Av, A, 0, i, A.Height(), 1);

            for(int j = 0; j < S; j++)
                P[j] = 1.0;

            typename std::list<_CWT_t>::const_iterator it;
            int qc = 0;
            for(it = _cwts.begin(); it != _cwts.end(); it++, qc++) {
                const _CWT_t &C = *it;
                C.apply(Av, W, columnwise_tag());
                El::Scale((value_type)std::sqrt(data_type::_gamma), W);
                W.Update(data_type::_hash_idx[qc], 0,
                    std::sqrt(data_type::_c) * data_type::_hash_val[qc]);
                internal::fftw<value_type>::executeffun(_fftw_fplan, W.Buffer(),
                    reinterpret_cast<_fftw_complex_t*>(FW));
                for(int j = 0; j < S; j++)
                    P[j] *= FW[j];
            }

            // In FFTW, both fft and ifft are not scaled.
            // That is norm(ifft(fft(x)) = norm(x) * #els(x).
            for(int j = 0; j < S; j++)
                P[j] /= (value_type)S;

            El::View(SAv, sketch_of_A, 0, i, A.Height(), 1);
            internal::fftw<value_type>::executebfun(_fftw_bplan,
                reinterpret_cast<_fftw_complex_t*>(P), SAv.Buffer());
        }

        delete[] FW;
        delete[] P;

        }
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        // TODO verify sizes etc.
        const int S = data_type::_S;
        const int N = data_type::_N;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel
#       endif
        {

        matrix_type W(S, 1);
        matrix_type ASv, SATv(S, 1);

        matrix_type Av, ATv;

        std::complex<value_type> *FW = new std::complex<value_type>[S];
        std::complex<value_type> *P = new std::complex<value_type>[S];

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp for
#       endif
        for(int i = 0; i < A.Height(); i++) {
            El::LockedView(Av, A, i, 0, 1, A.Width());
            El::Transpose(Av, ATv);

            for(int j = 0; j < S; j++)
                P[j] = 1.0;

            typename std::list<_CWT_t>::const_iterator it;
            int qc = 0;
            for(it = _cwts.begin(); it != _cwts.end(); it++, qc++) {
                const _CWT_t &C = *it;
                C.apply(ATv, W, columnwise_tag());
                El::Scale((value_type)std::sqrt(data_type::_gamma), W);
                W.Update(data_type::_hash_idx[qc], 0,
                    std::sqrt(data_type::_c) * data_type::_hash_val[qc]);
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
            El::View(ASv, sketch_of_A, i, 0, 1, sketch_of_A.Width());
            El::Transpose(SATv, ASv);
        }

        delete[] FW;
        delete[] P;

        }
    }

    int get_N() const { return data_type::_N; } /**< Get input dimesion. */
    int get_S() const { return data_type::_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

protected:

    typedef CWT_t<matrix_type, output_matrix_type> _CWT_t;

    typedef typename internal::fftw<value_type>::complex_t _fftw_complex_t;
    typedef typename internal::fftw<value_type>::plan_t _fftw_plan_t;

    _fftw_plan_t _fftw_fplan, _fftw_bplan;
    std::list<_CWT_t> _cwts;

    void build_internal() {
        int S = data_type::_S;

        for(typename std::list<CWT_data_t>::iterator it =
                data_type::_cwts_data.begin();
            it != data_type::_cwts_data.end(); it++)
            _cwts.push_back(_CWT_t(*it));

        value_type *dtmp = new value_type[S];
        std::complex<value_type> *ctmp = new std::complex<value_type>[S];

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp critical
#       endif
        {

        _fftw_fplan = internal::fftw<value_type>::fplanfun(S, dtmp,
            reinterpret_cast<_fftw_complex_t*>(ctmp),
            FFTW_UNALIGNED | FFTW_ESTIMATE);
        _fftw_bplan = internal::fftw<value_type>::bplanfun(S,
            reinterpret_cast<_fftw_complex_t*>(ctmp), dtmp,
            FFTW_UNALIGNED | FFTW_ESTIMATE);

        }

        delete[] dtmp;
        delete[] ctmp;

    }
};

/**
 * Specialization for sparse local to local.
 */
template<typename ValueType>
struct PPT_t <
    base::sparse_matrix_t<ValueType>,
    El::Matrix<ValueType> > :
        public PPT_data_t,
        virtual public sketch_transform_t<base::sparse_matrix_t<ValueType>,
                                          El::Matrix<ValueType> >{

    typedef ValueType value_type;
    typedef base::sparse_matrix_t<value_type> matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;

    typedef PPT_data_t data_type;
    typedef data_type::params_t params_t;

    PPT_t(int N, int S, int q, double c, double gamma, base::context_t& context)
        : data_type (N, S, q, c, gamma, context)  {

        build_internal();
    }

    PPT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context)  {

        build_internal();
    }

    PPT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

        build_internal();
    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    PPT_t(const PPT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

        build_internal();
    }

    PPT_t(const data_type& other_data)
        : data_type(other_data) {

        build_internal();
    }

    ~PPT_t() {
        internal::fftw<value_type>::destroyfun(_fftw_fplan);
        internal::fftw<value_type>::destroyfun(_fftw_bplan);
    }
    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {

        // TODO verify sizes etc.
        // TODO I am not sure this implementation is the most efficient
        //      for sparse matrices. Maybe you want to do the CWT right on
        //      start?

        const int S = data_type::_S;
        const int N = data_type::_N;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel
#       endif
        {

        output_matrix_type W(S, 1);
        output_matrix_type SAv;

        std::complex<value_type> *FW = new std::complex<value_type>[S];
        std::complex<value_type> *P = new std::complex<value_type>[S];

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp for
#       endif
        for(int i = 0; i < base::Width(A); i++) {
            const matrix_type Av = base::ColumnView(A, i, 1);

            for(int j = 0; j < S; j++)
                P[j] = 1.0;

            typename std::list<_CWT_t>::const_iterator it;
            int qc = 0;
            for(it = _cwts.begin(); it != _cwts.end(); it++, qc++) {
                const _CWT_t &C = *it;
                C.apply(Av, W, columnwise_tag());
                El::Scale((value_type)std::sqrt(data_type::_gamma), W);
                W.Update(data_type::_hash_idx[qc], 0,
                    std::sqrt(data_type::_c) * data_type::_hash_val[qc]);
                internal::fftw<value_type>::executeffun(_fftw_fplan, W.Buffer(),
                    reinterpret_cast<_fftw_complex_t*>(FW));
                for(int j = 0; j < S; j++)
                    P[j] *= FW[j];
            }

            // In FFTW, both fft and ifft are not scaled.
            // That is norm(ifft(fft(x)) = norm(x) * #els(x).
            for(int j = 0; j < S; j++)
                P[j] /= (value_type)S;

            El::View(SAv, sketch_of_A, 0, i, base::Height(A), 1);
            internal::fftw<value_type>::executebfun(_fftw_bplan,
                reinterpret_cast<_fftw_complex_t*>(P), SAv.Buffer());
        }

        delete[] FW;
        delete[] P;

        }
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
        // TODO perhaps not the best way to do this...
        matrix_type AT;
        base::Transpose(A, AT);
        output_matrix_type SAT(sketch_of_A.Width(), sketch_of_A.Height());
        apply(AT, SAT, columnwise_tag());
        El::Transpose(SAT, sketch_of_A);
    }

    int get_N() const { return data_type::_N; } /**< Get input dimesion. */
    int get_S() const { return data_type::_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

protected:

    typedef CWT_t<matrix_type, output_matrix_type> _CWT_t;

    typedef typename internal::fftw<value_type>::complex_t _fftw_complex_t;
    typedef typename internal::fftw<value_type>::plan_t _fftw_plan_t;

    _fftw_plan_t _fftw_fplan, _fftw_bplan;
    std::list<_CWT_t> _cwts;

    void build_internal() {
        int S = data_type::_S;

        for(typename std::list<CWT_data_t>::iterator it =
                data_type::_cwts_data.begin();
            it != data_type::_cwts_data.end(); it++)
            _cwts.push_back(_CWT_t(*it));

        value_type *dtmp = new value_type[S];
        std::complex<value_type> *ctmp = new std::complex<value_type>[S];

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp critical
#       endif
        {

        _fftw_fplan = internal::fftw<value_type>::fplanfun(S, dtmp,
            reinterpret_cast<_fftw_complex_t*>(ctmp),
            FFTW_UNALIGNED | FFTW_ESTIMATE);
        _fftw_bplan = internal::fftw<value_type>::bplanfun(S,
            reinterpret_cast<_fftw_complex_t*>(ctmp), dtmp,
            FFTW_UNALIGNED | FFTW_ESTIMATE);

        }

        delete[] dtmp;
        delete[] ctmp;
    }
};

/**
 * Specialization [STAR, STAR] to same distribution.
 */
template <typename ValueType>
struct PPT_t <
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR> > :
        public PPT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> output_matrix_type;
    typedef PPT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    PPT_t(const PPT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    PPT_t(const data_type& other_data)
        : data_type(other_data), _local(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        // Just a local operation on the Matrix
        _local.apply(A.LockedMatrix(), sketch_of_A.Matrix(), dimension);
    }

private:

    const PPT_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [CIRC, CIRC] to same distribution.
 */
template <typename ValueType>
struct PPT_t <
    El::DistMatrix<ValueType, El::CIRC, El::CIRC>,
    El::DistMatrix<ValueType, El::CIRC, El::CIRC> > :
        public PPT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::CIRC, El::CIRC> matrix_type;
    typedef El::DistMatrix<value_type, El::CIRC, El::CIRC> output_matrix_type;
    typedef PPT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    PPT_t(const PPT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    PPT_t(const data_type& other_data)
        : data_type(other_data), _local(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        // TODO do we allow different communicators and different roots?

        // If on root: Just a local operation on the Matrix
        if (skylark::utility::get_communicator(A).rank() == 0)
            _local.apply(A.LockedMatrix(), sketch_of_A.Matrix(), dimension);
    }

private:

    const PPT_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [VC/VR, STAR] to same distribution.
 */
template <typename ValueType, El::Distribution ColDist>
struct PPT_t <
    El::DistMatrix<ValueType, ColDist, El::STAR>,
    El::DistMatrix<ValueType, ColDist, El::STAR> > :
        public PPT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, ColDist, El::STAR> matrix_type;
    typedef El::DistMatrix<value_type, ColDist, El::STAR> output_matrix_type;
    typedef PPT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    PPT_t(const PPT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    PPT_t(const data_type& other_data)
        : data_type(other_data), _local(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        switch (ColDist) {
        case El::VR:
        case El::VC:
            try {
                apply_impl_vdist (A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    base::elemental_exception()
                        << base::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    base::mpi_exception()
                        << base::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                base::unsupported_matrix_distribution() );

        }
    }

private:
    /**
     * Apply the sketching transform on A and write to sketch_of_A.
     * Implementation for columnwise.
     */
    void apply_impl_vdist(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {

        // Naive implementation: tranpose and uses the columnwise implementation
        // Can we do better?
        matrix_type A_t(A.Grid());
        El::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height(), sketch_of_A.Grid());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::rowwise_tag());
        El::Transpose(sketch_of_A_t, sketch_of_A);
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // Just a local operation on the Matrix
        _local.apply(A.LockedMatrix(), sketch_of_A.Matrix(), tag);
    }

private:

    const PPT_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [STAR, VC/VR] to same distribution.
 */
template <typename ValueType, El::Distribution RowDist>
struct PPT_t <
    El::DistMatrix<ValueType, El::STAR, RowDist>,
    El::DistMatrix<ValueType, El::STAR, RowDist> > :
        public PPT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::STAR, RowDist> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, RowDist> output_matrix_type;
    typedef PPT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    PPT_t(const PPT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    PPT_t(const data_type& other_data)
        : data_type(other_data), _local(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        switch (RowDist) {
        case El::VR:
        case El::VC:
            try {
                apply_impl_vdist (A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    base::elemental_exception()
                        << base::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    base::mpi_exception()
                        << base::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                base::unsupported_matrix_distribution() );

        }
    }

private:
    /**
     * Apply the sketching transform on A and write to sketch_of_A.
     * Implementation for columnwise.
     */
    void apply_impl_vdist(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {


        // Just a local operation on the Matrix
        _local.apply(A.LockedMatrix(), sketch_of_A.Matrix(), tag);
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // Naive implementation: tranpose and uses the columnwise implementation
        // Can we do better?
        matrix_type A_t(A.Grid());
        El::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height(), sketch_of_A.Grid());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::rowwise_tag());
        El::Transpose(sketch_of_A_t, sketch_of_A);
    }

private:

    const PPT_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};


} } /** namespace skylark::sketch */

#endif // SKYLARK_HAVE_FFTW || SKYLARK_HAVE_FFTWF

#endif // PPT_ELEMENTAL_HPP
