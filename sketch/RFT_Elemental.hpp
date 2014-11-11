#ifndef SKYLARK_RFT_ELEMENTAL_HPP
#define SKYLARK_RFT_ELEMENTAL_HPP

namespace skylark {
namespace sketch {

/**
 * Specialization for local output
 */
template <typename ValueType,
          typename InputType,
          template <typename> class KernelDistribution>
struct RFT_t <
    InputType,
    elem::Matrix<ValueType>,
    KernelDistribution> :
        public RFT_data_t<KernelDistribution> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef RFT_data_t<KernelDistribution> data_type;

private:
    typedef dense_transform_t <matrix_type, output_matrix_type,
                               typename data_type::accessor_type> underlying_t;


protected:
    /**
     * Regular constructor - Allow creation only by subclasses
     */
    RFT_t (int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

public:
    /**
     * Copy constructor
     */
    RFT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    RFT_t(const data_type& other_data)
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

        // TODO verify sizes etc.
        underlying_t underlying(*data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for(int j = 0; j < base::Width(A); j++)
            for(int i = 0; i < data_type::_S; i++) {
                value_type x = sketch_of_A.Get(i, j);
                x += data_type::_shifts[i];

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

                x = data_type::_outscale * x;
                sketch_of_A.Set(i, j, x);
            }
    }

    /**
      * Apply the sketching transform on A and write to  sketch_of_A.
      * Implementation rowwise.
      */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // TODO verify sizes etc.
        underlying_t underlying(*data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for(int j = 0; j < data_type::_S; j++)
            for(int i = 0; i < base::Height(A); i++) {
                value_type x = sketch_of_A.Get(i, j);
                x += data_type::_shifts[j];

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

                x = data_type::_outscale * x;
                sketch_of_A.Set(i, j, x);
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
struct RFT_t <
    InputType,
    elem::DistMatrix<ValueType, OC, OR>,
    KernelDistribution> :
        public RFT_data_t<KernelDistribution> {
    // Typedef value, matrix, transform, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef InputType matrix_type;
    typedef elem::DistMatrix<value_type, OC, OR> output_matrix_type;

    typedef RFT_data_t<KernelDistribution> data_type;

private:
    typedef dense_transform_t <matrix_type, output_matrix_type,
                               typename data_type::accessor_type> underlying_t;

protected:

    // Allow only creation by subclasses.

    /**
     * Regular constructor -- Allow only creation by subclasses
     */
    RFT_t (int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

public:

    /**
     * Copy constructor
     */
    RFT_t(const RFT_t<matrix_type,
                      output_matrix_type,
                      KernelDistribution>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    RFT_t(const data_type& other_data)
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
            apply_impl (A, sketch_of_A, dimension);
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
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for columnwise.
     */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {

        underlying_t underlying(*data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);

        elem::Matrix<value_type> &SAl = sketch_of_A.Matrix();
        size_t col_shift = sketch_of_A.ColShift();
        size_t col_stride = sketch_of_A.ColStride();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for(size_t j = 0; j < base::Width(SAl); j++)
            for(size_t i = 0; i < base::Height(SAl); i++) {
                value_type x = SAl.Get(i, j);
                x += data_type::_shifts[col_shift + i * col_stride];

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

                x = data_type::_outscale * x;
                SAl.Set(i, j, x);
            }
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for rowwise.
     */
    void apply_impl(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        // TODO verify sizes etc.
        underlying_t underlying(*data_type::_underlying_data);
        underlying.apply(A, sketch_of_A, tag);

        elem::Matrix<value_type> &SAl = sketch_of_A.Matrix();
        size_t row_shift = sketch_of_A.RowShift();
        size_t row_stride = sketch_of_A.RowStride();

#       if SKYLARK_HAVE_OPENMP
#       pragma omp parallel for collapse(2)
#       endif
        for(size_t j = 0; j < base::Width(SAl); j++)
            for(size_t i = 0; i < base::Height(SAl); i++) {
                value_type x = SAl.Get(i, j);
                x += data_type::_shifts[row_shift + j * row_stride];

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

                x = data_type::_outscale * x;
                SAl.Set(i, j, x);
            }
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_RFT_ELEMENTAL_HPP
