#ifndef SKYLARK_FRFT_ELEMENTAL_HPP
#define SKYLARK_FRFT_ELEMENTAL_HPP

namespace skylark {
namespace sketch {

#if SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT

/**
 * Specialization local input (sparse of dense), local output.
 * InputType should either be elem::Matrix, or base:spare_matrix_t.
 */
template <typename ValueType,
          template <typename> class InputType>
struct FastRFT_t <
    InputType<ValueType>,
    elem::Matrix<ValueType> > :
        public FastRFT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType<value_type> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef FastRFT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _fut(data_type::_N) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const data_type& other_data)
        : data_type(other_data), _fut(data_type::_N) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl(A, sketch_of_A, dimension);
        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (
                base::elemental_exception()
                    << base::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (
                base::mpi_exception()
                    << base::error_msg(e.what()) );
        }
    }

private:
    /**
     * Apply the sketching transform on A and write to sketch_of_A.
     * Implementation for columnwise.
     */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel
#       endif
        {
        output_matrix_type W(data_type::_NB, 1);
        double *w = W.Buffer();

        output_matrix_type Ac(data_type::_NB, 1);
        double *ac = Ac.Buffer();

        output_matrix_type Acv;
        elem::View(Acv, Ac, 0, 0, data_type::_N, 1);

        double *sa = sketch_of_A.Buffer();
        int ldsa = sketch_of_A.LDim();

        value_type scal =
            std::sqrt(data_type::_NB) * _fut.scale();

        output_matrix_type B(data_type::_NB, 1), G(data_type::_NB, 1);
        output_matrix_type Sm(data_type::_NB, 1);

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp for
#       endif
        for(int c = 0; c < base::Width(A); c++) {
            const matrix_type Acs = base::ColumnView(A, c, 1);
            base::DenseCopy(Acs, Acv);
            std::fill(ac + data_type::_N, ac + data_type::_NB, 0);

            for(int i = 0; i < data_type::numblks; i++) {

                int s = i * data_type::_NB;
                int e = std::min(s + data_type::_NB,  data_type::_S);

                // Set the local values of B, G and S
                for(int j = 0; j < data_type::_NB; j++) {
                    B.Set(j, 0, data_type::B[i * data_type::_NB + j]);
                    G.Set(j, 0, scal * data_type::G[i * data_type::_NB + j]);
                    Sm.Set(j, 0, scal * data_type::Sm[i * data_type::_NB + j]);
                }

                W = Ac;

                elem::DiagonalScale(elem::LEFT, elem::NORMAL, B, W);
                _fut.apply(W, tag);
                for(int l = 0; l < data_type::_NB - 1; l++) {
                    int idx1 = data_type::_NB - 1 - l;
                    int idx2 = data_type::P[i * (data_type::_NB - 1) + l];
                    std::swap(w[idx1], w[idx2]);
                }
                elem::DiagonalScale(elem::LEFT, elem::NORMAL, G, W);
                _fut.apply(W, tag);
                elem::DiagonalScale(elem::LEFT, elem::NORMAL, Sm, W);

                double *sac = sa + ldsa * c;
                for(int l = s; l < e; l++) {
                    value_type x = w[l - s];
                    x += data_type::shifts[l];

#                   ifdef SKYLARK_EXACT_COSINE
                    x = std::cos(x);
#                   else
                    // x = std::cos(x) is slow
                    // Instead use low-accuracy approximation
                    if (x < -3.14159265) x += 6.28318531;
                    else if (x >  3.14159265) x -= 6.28318531;
                    x += 1.57079632;
                    if (x >  3.14159265)
                        x -= 6.28318531;
                    x = (x < 0) ?
                        1.27323954 * x + 0.405284735 * x * x :
                        1.27323954 * x - 0.405284735 * x * x;
#                   endif

                    x = data_type::scale * x;
                    sac[l - s] = x;
                }
            }
        }

        }
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // TODO this version is really bad: it completely densifies the matrix
        //      on the begining.
        // TODO this version does not work with _NB and N
        // TODO this version is not as optimized as the columnwise version.

        // Create a work array W
        output_matrix_type W(base::Height(A), base::Width(A));

        output_matrix_type B(data_type::_N, 1), G(data_type::_N, 1);
        output_matrix_type Sm(data_type::_N, 1);
        for(int i = 0; i < data_type::numblks; i++) {
            int s = i * data_type::_N;
            int e = std::min(s + data_type::_N, data_type::_S);

            base::DenseCopy(A, W);

            // Set the local values of B, G and S
            value_type scal =
                std::sqrt(data_type::_N) * _fut.scale();
            for(int j = 0; j < data_type::_N; j++) {
                B.Set(j, 0, data_type::B[i * data_type::_N + j]);
                G.Set(j, 0, scal * data_type::G[i * data_type::_N + j]);
                Sm.Set(j, 0, scal * data_type::Sm[i * data_type::_N + j]);
            }

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, B, W);

            _fut.apply(W, tag);

            double *w = W.Buffer();
            for(int c = 0; c < base::Height(W); c++)
                for(int l = 0; l < data_type::_N - 1; l++) {
                    int idx1 = c + (data_type::_N - 1 - l) * W.LDim();
                    int idx2 = c  +
                        (data_type::P[i * (data_type::_N - 1) + l]) * W.LDim();
                    std::swap(w[idx1], w[idx2]);
                }

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, G, W);

            _fut.apply(W, tag);

            elem::DiagonalScale(elem::RIGHT, elem::NORMAL, Sm, W);

            // Copy that part to the output
            output_matrix_type view_sketch_of_A;
            elem::View(view_sketch_of_A, sketch_of_A, 0, s,
                base::Height(A), e - s);
            output_matrix_type view_W;
            elem::View(view_W, W, 0, 0, base::Height(A), e - s);
            view_sketch_of_A = view_W;
        }

        for(int j = 0; j < data_type::_S; j++)
            for(int i = 0; i < base::Height(A); i++) {
                value_type x = sketch_of_A.Get(i, j);
                x += data_type::shifts[j];

#               ifdef SKYLARK_EXACT_COSINE
                x = std::cos(x);
#               else
                // x = std::cos(x) is slow
                // Instead use low-accuracy approximation
                if (x < -3.14159265) x += 6.28318531;
                else if (x >  3.14159265) x -= 6.28318531;
                x += 1.57079632;
                if (x >  3.14159265)
                    x -= 6.28318531;
                x = (x < 0) ?
                    1.27323954 * x + 0.405284735 * x * x :
                    1.27323954 * x - 0.405284735 * x * x;
#               endif

                x = data_type::scale * x;
                sketch_of_A.Set(i, j, x);
            }
    }

private:

#ifdef SKYLARK_HAVE_FFTW
    typename fft_futs<ValueType>::DCT_t _fut;
#elif SKYLARK_HAVE_SPIRALWHT
    WHT_t<double> _fut;
#endif

};

/**
 * Specialization [STAR, STAR] to same distribution.
 */
template <typename ValueType>
struct FastRFT_t <
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR> > :
        public FastRFT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, elem::STAR, elem::STAR> output_matrix_type;
    typedef FastRFT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const data_type& other_data)
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

    const FastRFT_t<elem::Matrix<value_type>, elem::Matrix<value_type> > _local;
};

/**
 * Specialization [CIRC, CIRC] to same distribution.
 */
template <typename ValueType>
struct FastRFT_t <
    elem::DistMatrix<ValueType, elem::CIRC, elem::CIRC>,
    elem::DistMatrix<ValueType, elem::CIRC, elem::CIRC> > :
        public FastRFT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::CIRC, elem::CIRC> matrix_type;
    typedef elem::DistMatrix<value_type, elem::CIRC, elem::CIRC> output_matrix_type;
    typedef FastRFT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const data_type& other_data)
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

    const FastRFT_t<elem::Matrix<value_type>, elem::Matrix<value_type> > _local;
};

/**
 * Specialization [VC/VR, STAR] to same distribution.
 */
template <typename ValueType, elem::Distribution ColDist>
struct FastRFT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, ColDist, elem::STAR> > :
        public FastRFT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> output_matrix_type;
    typedef FastRFT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const data_type& other_data)
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
        case elem::VR:
        case elem::VC:
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
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height(), sketch_of_A.Grid());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::rowwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
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

    const FastRFT_t<elem::Matrix<value_type>, elem::Matrix<value_type> > _local;
};

/**
 * Specialization [STAR, VC/VR] to same distribution.
 */
template <typename ValueType, elem::Distribution RowDist>
struct FastRFT_t <
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    elem::DistMatrix<ValueType, elem::STAR, RowDist> > :
        public FastRFT_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> matrix_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> output_matrix_type;
    typedef FastRFT_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    FastRFT_t(const FastRFT_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    FastRFT_t(const data_type& other_data)
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
        case elem::VR:
        case elem::VC:
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
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height(), sketch_of_A.Grid());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::rowwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
    }

private:

    const FastRFT_t<elem::Matrix<value_type>, elem::Matrix<value_type> > _local;
};

#endif // SKYLARK_HAVE_FFTW || SKYLARK_HAVE_SPIRALWHT

} } /** namespace skylark::sketch */

#endif // SKYLARK_FRFT_ELEMENTAL_HPP
