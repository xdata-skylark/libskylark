#ifndef SKYLARK_RLT_ELEMENTAL_HPP
#define SKYLARK_RLT_ELEMENTAL_HPP

namespace skylark {
namespace sketch {

/**
 * Specialization for local output
 */
template <typename ValueType,
          typename InputType,
          template <typename> class KernelDistribution>
struct RLT_t <
    InputType,
    elem::Matrix<ValueType>,
    KernelDistribution> :
        public RLT_data_t<KernelDistribution> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef RLT_data_t<KernelDistribution> data_type;

private:
    typedef dense_transform_t <matrix_type, output_matrix_type,
                               typename data_type::accessor_type> underlying_t;

protected:
    /**
     * Regular constructor - allow creation only by subclasses
     */
    RLT_t (int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

public:
    /**
     * Copy constructor
     */
    RLT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    RLT_t(const data_type& other_data)
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
          elem::Distribution OC, elem::Distribution OR,
          template <typename> class KernelDistribution>
struct RLT_t <
    InputType,
    elem::DistMatrix<ValueType, OC, OR>,
    KernelDistribution> :
        public RLT_data_t<KernelDistribution> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType matrix_type;
    typedef elem::DistMatrix<value_type, OC, OR> output_matrix_type;
    typedef RLT_data_t<KernelDistribution> data_type;

private:
    typedef dense_transform_t <matrix_type, output_matrix_type,
                               typename data_type::accessor_type> underlying_t;


protected:
    /**
     * Regular constructor - allow creation only by subclasses
     */
    RLT_t (int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

public:
    /**
     * Copy constructor
     */
    RLT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    RLT_t(const data_type& other_data)
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

            elem::Matrix<value_type> &SAl = sketch_of_A.Matrix();

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

#endif // SKYLARK_RLT_ELEMENTAL_HPP
