#ifndef SKYLARK_UST_ELEMENTAL_HPP
#define SKYLARK_UST_ELEMENTAL_HPP

namespace skylark { namespace sketch {

/**
 * Specialization for local to local.
 */
template<typename ValueType>
struct UST_t <
    El::Matrix<ValueType>,
    El::Matrix<ValueType> > :
        public UST_data_t,
        virtual public sketch_transform_t<El::Matrix<ValueType>,
                                          El::Matrix<ValueType> >{

    typedef ValueType value_type;
    typedef El::Matrix<value_type> matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;

    typedef UST_data_t data_type;
    typedef data_type::params_t params_t;

    UST_t(int N, int S, bool replace, base::context_t& context)
        : data_type (N, S, replace, context)  {

     }

    UST_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context)  {

    }

    UST_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    UST_t(const UST_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    UST_t(const data_type& other_data)
        : data_type(other_data) {

    }

    ~UST_t() {
    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {

        const value_type *a = A.LockedBuffer();
        El::Int lda = A.LDim();
        value_type *sa = sketch_of_A.Buffer();
        El::Int ldsa = sketch_of_A.LDim();

        for (El::Int j = 0; j < A.Width(); j++)
            for (El::Int i = 0; i < data_type::_S; i++)
                sa[j * ldsa + i] = a[j * lda + data_type::_samples[i]];
    }

    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {

        const value_type *a = A.LockedBuffer();
        El::Int lda = A.LDim();
        value_type *sa = sketch_of_A.Buffer();
        El::Int ldsa = sketch_of_A.LDim();

        for (El::Int j = 0; j < data_type::_S; j++)
            for (El::Int i = 0; i < A.Height(); i++)
                sa[j * ldsa + i] = a[data_type::_samples[j] * lda + i];
    }

    int get_N() const { return data_type::_N; } /**< Get input dimesion. */
    int get_S() const { return data_type::_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }


};

/**
 * Specialization [STAR, STAR] to same distribution.
 */
template <typename ValueType>
struct UST_t <
    El::DistMatrix<ValueType, El::STAR, El::STAR>,
    El::DistMatrix<ValueType, El::STAR, El::STAR> > :
        public UST_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, El::STAR> output_matrix_type;
    typedef UST_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    UST_t(const UST_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    UST_t(const data_type& other_data)
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

    const UST_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [CIRC, CIRC] to same distribution.
 */
template <typename ValueType>
struct UST_t <
    El::DistMatrix<ValueType, El::CIRC, El::CIRC>,
    El::DistMatrix<ValueType, El::CIRC, El::CIRC> > :
        public UST_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::CIRC, El::CIRC> matrix_type;
    typedef El::DistMatrix<value_type, El::CIRC, El::CIRC> output_matrix_type;
    typedef UST_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    UST_t(const UST_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    UST_t(const data_type& other_data)
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

    const UST_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [VC/VR, STAR] to same distribution.
 */
template <typename ValueType, El::Distribution ColDist>
struct UST_t <
    El::DistMatrix<ValueType, ColDist, El::STAR>,
    El::DistMatrix<ValueType, ColDist, El::STAR> > :
        public UST_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, ColDist, El::STAR> matrix_type;
    typedef El::DistMatrix<value_type, ColDist, El::STAR> output_matrix_type;
    typedef UST_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    UST_t(const UST_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    UST_t(const data_type& other_data)
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

    const UST_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [STAR, VC/VR] to same distribution.
 */
template <typename ValueType, El::Distribution RowDist>
struct UST_t <
    El::DistMatrix<ValueType, El::STAR, RowDist>,
    El::DistMatrix<ValueType, El::STAR, RowDist> > :
        public UST_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, El::STAR, RowDist> matrix_type;
    typedef El::DistMatrix<value_type, El::STAR, RowDist> output_matrix_type;
    typedef UST_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    UST_t(const UST_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other), _local(other) {

    }

    /**
     * Constructor from data
     */
    UST_t(const data_type& other_data)
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
            skylark::sketch::columnwise_tag());
        El::Transpose(sketch_of_A_t, sketch_of_A);
    }

private:

    const UST_t<El::Matrix<value_type>, El::Matrix<value_type> > _local;
};

/**
 * Specialization [MC, MR] to [MC, MR].
 */
template <typename ValueType>
struct UST_t <
    El::DistMatrix<ValueType>,
    El::DistMatrix<ValueType> > :
        public UST_data_t {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type> matrix_type;
    typedef El::DistMatrix<value_type> output_matrix_type;
    typedef UST_data_t data_type;

public:

    // No regular contructor, since need to be subclassed.

    /**
     * Copy constructor
     */
    UST_t(const UST_t<matrix_type,
                      output_matrix_type>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    UST_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply the sketching transform on A and write to sketch_of_A.
     * Implementation for columnwise.
     */
    void apply(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {

        typedef El::DistMatrix<value_type, El::STAR, El::VR> redis_type;

        // Naive implementation: redistribute, sketch and redistribute
        UST_t<redis_type, redis_type> S1(*this);
        redis_type A1 = A;
        redis_type SA1(_S, A1.Width());
        S1.apply(A1, SA1, tag);
        sketch_of_A = SA1;
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        typedef El::DistMatrix<value_type, El::VC, El::STAR> redis_type;

        // Naive implementation: redistribute, sketch and redistribute
        UST_t<redis_type, redis_type> S1(*this);
        redis_type A1 = A;
        redis_type SA1(A1.Height(), _S);
        S1.apply(A1, SA1, tag);
        sketch_of_A = SA1;
    }
};

} } /** namespace skylark::sketch */

#endif // UST_ELEMENTAL_HPP
