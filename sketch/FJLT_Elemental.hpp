#ifndef SKYLARK_FJLT_ELEMENTAL_HPP
#define SKYLARK_FJLT_ELEMENTAL_HPP

#include "../utility/get_communicator.hpp"

namespace skylark { namespace sketch {

/**
 * Specialization for [*, SOMETHING]
 */
template <typename ValueType, elem::Distribution ColDist>
struct FJLT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType> > :
        public FJLT_data_t,
        virtual public sketch_transform_t<elem::DistMatrix<ValueType,
                                                           ColDist,
                                                           elem::STAR>,
                                          elem::Matrix<ValueType> > {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef elem::DistMatrix<ValueType,
                             elem::STAR, ColDist> intermediate_type;
    typedef fft_futs<double>::DCT_t transform_type;
    typedef utility::rademacher_distribution_t<value_type>
    underlying_value_distribution_type;

    typedef FJLT_data_t data_type;
    typedef data_type::params_t params_t;

protected:
    typedef RFUT_t<intermediate_type,
                   transform_type,
                   underlying_value_distribution_type> underlying_type;

public:
    FJLT_t(int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    FJLT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context) {

    }

    FJLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FJLT_t(const FJLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    FJLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
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


    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_A,
                    skylark::sketch::columnwise_tag) const {

        // Rearrange the matrix to fit the underlying transform
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply the underlying transform
        underlying_type underlying(*data_type::underlying_data);
        underlying.apply(inter_A, inter_A,
            skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(data_type::_S,
            inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)data_type::_N / (double)data_type::_S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < data_type::_S; i++) {
                int row = data_type::samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        // get communicator from matrix
        boost::mpi::communicator comm = skylark::utility::get_communicator(A);
        skylark::utility::collect_dist_matrix(comm, comm.rank() == 0,
            dist_sketch_A, sketch_A);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and rowwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_of_A,
                    skylark::sketch::rowwise_tag) const {

        // TODO This is a quick&dirty hack - uses the columnwise implementation.
        matrix_type A_t(A.Grid());
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::columnwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
     }
};

/**
 * Specialization for [*, SOMETHING] to [*, *]
 */
template <typename ValueType, elem::Distribution ColDist>
struct FJLT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR> > :
        public FJLT_data_t,
        virtual public sketch_transform_t<elem::DistMatrix<ValueType,
                                                           ColDist,
                                                           elem::STAR>,
                                          elem::DistMatrix<ValueType,
                                                           elem::STAR,
                                                           elem::STAR> > {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, elem::STAR, elem::STAR>
    output_matrix_type;
    typedef elem::DistMatrix<ValueType,
                             elem::STAR, ColDist> intermediate_type;
    typedef fft_futs<double>::DCT_t transform_type;
    typedef utility::rademacher_distribution_t<value_type>
    underlying_value_distribution_type;

    typedef FJLT_data_t data_type;
    typedef data_type::params_t params_t;

protected:
    typedef RFUT_t<intermediate_type,
                   transform_type,
                   underlying_value_distribution_type> underlying_type;

public:

    FJLT_t(int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    FJLT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FJLT_t(const FJLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    FJLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    FJLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
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


    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_A,
                    skylark::sketch::columnwise_tag) const {

        // Rearrange the matrix to fit the underlying transform
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply the underlying transform
        underlying_type underlying(*data_type::underlying_data);
        underlying.apply(inter_A, inter_A,
            skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(data_type::_S,
            inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)data_type::_N / (double)data_type::_S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < data_type::_S; i++) {
                int row = data_type::samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        sketch_A = dist_sketch_A;
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and rowwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_of_A,
                    skylark::sketch::rowwise_tag) const {

        // TODO This is a quick&dirty hack - uses the columnwise implementation.
        matrix_type A_t(A.Grid());
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::columnwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
     }
};


    /////////////////////////////////////////////////////

/**
 * Specialization for [SOMETHING, *]
 */
template <typename ValueType, elem::Distribution RowDist>
struct FJLT_t <
  elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    elem::Matrix<ValueType> > :
        public FJLT_data_t,
        virtual public sketch_transform_t<elem::DistMatrix<ValueType,
                                                           elem::STAR,
                                                           RowDist>,
                                          elem::Matrix<ValueType> > {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef elem::DistMatrix<ValueType,
                             elem::STAR, RowDist> intermediate_type;
    typedef fft_futs<double>::DCT_t transform_type;
    typedef utility::rademacher_distribution_t<value_type>
    underlying_value_distribution_type;

    typedef FJLT_data_t data_type;
    typedef data_type::params_t params_t;

protected:
    typedef RFUT_t<intermediate_type,
                   transform_type,
                   underlying_value_distribution_type> underlying_type;

public:
    FJLT_t(int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    FJLT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context) {

    }

    FJLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FJLT_t(const FJLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    FJLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
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


    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [*, VR/VC] and columnwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_A,
                    skylark::sketch::columnwise_tag) const {

        // Rearrange the matrix to fit the underlying transform; simply copy
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply the underlying transform
        underlying_type underlying(*data_type::underlying_data);
        underlying.apply(inter_A, inter_A,
            skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(data_type::_S,
            inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)data_type::_N / (double)data_type::_S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < data_type::_S; i++) {
                int row = data_type::samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        // get communicator from matrix
        boost::mpi::communicator comm = skylark::utility::get_communicator(A);
        skylark::utility::collect_dist_matrix(comm, comm.rank() == 0,
            dist_sketch_A, sketch_A);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [*, VR/VC] and rowwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_of_A,
                    skylark::sketch::rowwise_tag) const {

        // TODO This is a quick&dirty hack - uses the columnwise implementation.
        matrix_type A_t(A.Grid());
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::columnwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
     }
};


/**
 * Specialization for [SOMETHING, *] to [*, *]
 */
template <typename ValueType, elem::Distribution RowDist>
struct FJLT_t <
    elem::DistMatrix<ValueType, elem::STAR, RowDist>,
    elem::DistMatrix<ValueType, elem::STAR, elem::STAR> > :
        public FJLT_data_t,
        virtual public sketch_transform_t<elem::DistMatrix<ValueType,
                                                           elem::STAR,
                                                           RowDist>,
                                          elem::DistMatrix<ValueType,
                                                           elem::STAR,
                                                           elem::STAR> > {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, elem::STAR, RowDist> matrix_type;
    typedef elem::DistMatrix<value_type, elem::STAR, elem::STAR>
    output_matrix_type;
    typedef elem::DistMatrix<ValueType,
                             elem::STAR, RowDist> intermediate_type;
    typedef fft_futs<double>::DCT_t transform_type;
    typedef utility::rademacher_distribution_t<value_type>
    underlying_value_distribution_type;

    typedef FJLT_data_t data_type;
    typedef data_type::params_t params_t;

protected:
    typedef RFUT_t<intermediate_type,
                   transform_type,
                   underlying_value_distribution_type> underlying_type;

public:

    FJLT_t(int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    FJLT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FJLT_t(const FJLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    FJLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    FJLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }

    /**
     * Apply columnwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                columnwise_tag dimension) const {
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


    /**
     * Apply rowwise the sketching transform that is described by the
     * the transform with output sketch_of_A.
     */
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                rowwise_tag dimension) const {
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

    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [*, VR/VC] and columnwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_A,
                    skylark::sketch::columnwise_tag) const {

        // Rearrange the matrix to fit the underlying transform; simply copy
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply the underlying transform
        underlying_type underlying(*data_type::underlying_data);
        underlying.apply(inter_A, inter_A,
            skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(data_type::_S,
            inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)data_type::_N / (double)data_type::_S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < data_type::_S; i++) {
                int row = data_type::samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        sketch_A = dist_sketch_A;
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and rowwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_of_A,
                    skylark::sketch::rowwise_tag) const {

        // TODO This is a quick&dirty hack - uses the columnwise implementation.
        matrix_type A_t(A.Grid());
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::columnwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
     }
};


/**
 * Specialization for [MC, MR] to [MC, MR]
 */
template <typename ValueType>
struct FJLT_t <
    elem::DistMatrix<ValueType>,
    elem::DistMatrix<ValueType> > :
        public FJLT_data_t,
        virtual public sketch_transform_t<elem::DistMatrix<ValueType>,
                                          elem::DistMatrix<ValueType> > {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type> matrix_type;
    typedef elem::DistMatrix<value_type>
    output_matrix_type;

    typedef elem::DistMatrix<ValueType,
                             elem::STAR, elem::VR>
    intermediate_type_star_rowdist;

    typedef elem::DistMatrix<ValueType,
                             elem::VC, elem::STAR>
    intermediate_type_coldist_star;

    typedef fft_futs<double>::DCT_t transform_type;
    typedef utility::rademacher_distribution_t<value_type>
    underlying_value_distribution_type;

    typedef FJLT_data_t data_type;
    typedef data_type::params_t params_t;

protected:
    typedef RFUT_t<intermediate_type_coldist_star,
                   transform_type,
                   underlying_value_distribution_type>
    underlying_type_coldist_star;

    typedef RFUT_t<intermediate_type_star_rowdist,
                   transform_type,
                   underlying_value_distribution_type>
    underlying_type_star_rowdist;

public:

    FJLT_t(int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    FJLT_t(int N, int S, const params_t& params, base::context_t& context)
        : data_type (N, S, params, context) {

    }

    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FJLT_t(const FJLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : data_type(other) {

    }

    FJLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    FJLT_t(const boost::property_tree::ptree &pt)
        : data_type(pt) {

    }


    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {
            apply_impl_dist(A, sketch_of_A, dimension);
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


    int get_N() const { return this->_N; } /**< Get input dimesion. */
    int get_S() const { return this->_S; } /**< Get output dimesion. */

    const sketch_transform_data_t* get_data() const { return this; }

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [MC, MR] and columnwise.
     */
    void apply_impl_dist(const matrix_type& A,
                    output_matrix_type& sketch_A,
                    skylark::sketch::columnwise_tag) const {

        // Rearrange the matrix to fit the underlying transform
        typedef intermediate_type_star_rowdist intermediate_type;
        typedef underlying_type_star_rowdist   underlying_type;
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply the underlying transform
        underlying_type underlying(*data_type::underlying_data);
        underlying.apply(inter_A, inter_A,
            skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(data_type::_S,
            inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)data_type::_N / (double)data_type::_S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < data_type::_S; i++) {
                int row = data_type::samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        sketch_A = dist_sketch_A;
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [MC, MR] and rowwise.
     *
     * XXX Avoid transposition by working on the underlying type/transform
     */
    void apply_impl_dist(const matrix_type& A,
                    output_matrix_type& sketch_of_A,
                    skylark::sketch::rowwise_tag) const {

        // TODO This is a quick&dirty hack - uses the columnwise implementation.
        matrix_type A_t(A.Grid());
        elem::Transpose(A, A_t);
        output_matrix_type sketch_of_A_t(sketch_of_A.Width(),
            sketch_of_A.Height());
        apply_impl_vdist(A_t, sketch_of_A_t,
            skylark::sketch::columnwise_tag());
        elem::Transpose(sketch_of_A_t, sketch_of_A);
     }
};


} } /** namespace skylark::sketch */

#endif // FJLT_ELEMENTAL_HPP
