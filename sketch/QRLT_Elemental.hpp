#ifndef SKYLARK_QRLT_ELEMENTAL_HPP
#define SKYLARK_QRLT_ELEMENTAL_HPP

namespace skylark {
namespace sketch {

/**
 * Specialization for local output
 */
template <typename ValueType,
          typename InputType,
          template <typename, typename> class KernelDistribution,
          template <typename> class QMCSequenceType>
struct QRLT_t <
    InputType,
    El::Matrix<ValueType>,
    KernelDistribution, QMCSequenceType> :
        public QRLT_data_t<KernelDistribution, QMCSequenceType> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType matrix_type;
    typedef El::Matrix<value_type> output_matrix_type;
    typedef QRLT_data_t<KernelDistribution, QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;

private:
    typedef dense_transform_t <matrix_type, output_matrix_type,
                               typename data_type::accessor_type> underlying_t;

protected:
    /**
     * Regular constructor - allow creation only by subclasses
     */
    QRLT_t (int N, int S,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type (N, S, sequence, skip, context) {

    }

public:
    /**
     * Copy constructor
     */
    QRLT_t(const QRLT_t<matrix_type,
        output_matrix_type,
        KernelDistribution, QMCSequenceType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    QRLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {

            // TODO verify sizes etc.
            underlying_t underlying(*data_type::_underlying_data);
            underlying.apply(A, sketch_of_A, dimension);

#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for collapse(2)
#           endif
            for(int j = 0; j < base::Width(sketch_of_A); j++)
                for(int i = 0; i < base::Height(sketch_of_A); i++) {
                    value_type val = sketch_of_A.Get(i, j);
                    sketch_of_A.Set(i, j,
                        data_type::_outscale * std::exp(-val));
                }

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
};

/**
 * Specialization for distributed output
 */
template <typename ValueType,
          typename InputType,
          El::Distribution OC, El::Distribution OR,
          template <typename, typename> class KernelDistribution,
          template <typename> class QMCSequenceType>
struct QRLT_t <
    InputType,
    El::DistMatrix<ValueType, OC, OR>,
    KernelDistribution, QMCSequenceType> :
        public QRLT_data_t<KernelDistribution, QMCSequenceType> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType matrix_type;
    typedef El::DistMatrix<value_type, OC, OR> output_matrix_type;

    typedef QRLT_data_t<KernelDistribution, QMCSequenceType> data_type;
    typedef typename data_type::sequence_type sequence_type;

private:
    typedef dense_transform_t <matrix_type, output_matrix_type,
                               typename data_type::accessor_type> underlying_t;


protected:
    /**
     * Regular constructor - allow creation only by subclasses
     */
    QRLT_t (int N, int S,
        const sequence_type& sequence, int skip, base::context_t& context)
        : data_type (N, S, sequence, skip, context) {

    }

public:
    /**
     * Copy constructor
     */
    QRLT_t(const QRLT_t<matrix_type,
        output_matrix_type,
        KernelDistribution, QMCSequenceType>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    QRLT_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {
        try {

            // TODO verify sizes etc.
            underlying_t underlying(*data_type::_underlying_data);
            underlying.apply(A, sketch_of_A, dimension);

            El::Matrix<value_type> &SAl = sketch_of_A.Matrix();

#           if SKYLARK_HAVE_OPENMP
#           pragma omp parallel for collapse(2)
#           endif
            for(int j = 0; j < base::Width(SAl); j++)
                for(int i = 0; i < base::Height(SAl); i++) {
                    value_type val = SAl.Get(i, j);
                    SAl.Set(i, j,
                        data_type::_outscale * std::exp(-val));
                }

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
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_QRLT_ELEMENTAL_HPP
