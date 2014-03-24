#ifndef SKYLARK_FJLT_ELEMENTAL_HPP
#define SKYLARK_FJLT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "FJLT_data.hpp"
#include "../utility/exception.hpp"

namespace skylark { namespace sketch {
/**
 * Specialization for [*, SOMETHING]
 */
template <typename ValueType, elem::Distribution ColDist>
struct FJLT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType> > :
        public FJLT_data_t<ValueType> {
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

protected:
    typedef FJLT_data_t<value_type> base_data_t;
    typedef RFUT_t<intermediate_type,
                   transform_type,
                   underlying_value_distribution_type> underlying_type;

public:
    /**
     * Regular constructor
     */
    FJLT_t(int N, int S, skylark::sketch::context_t& context)
        : base_data_t (N, S, context) {

    }

    /**
     * Copy constructor
     */
    template <typename OtherInputMatrixType,
              typename OtherOutputMatrixType>
    FJLT_t(const FJLT_t<OtherInputMatrixType, OtherOutputMatrixType>& other)
        : base_data_t(other) {

    }

    /**
     * Constructor from data
     */
    FJLT_t(const base_data_t& other_data)
        : base_data_t(other_data) {

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
                    utility::elemental_exception()
                        << utility::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    utility::mpi_exception()
                        << utility::error_msg(e.what()) );
            }

            break;

        default:
            SKYLARK_THROW_EXCEPTION (
                utility::unsupported_matrix_distribution() );

        }
    }

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
        underlying_type underlying(base_data_t::underlying_data);
        underlying.apply(inter_A, inter_A,
            skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(base_data_t::S,
            inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)base_data_t::N / (double)base_data_t::S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < base_data_t::S; i++) {
                int row = base_data_t::samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        skylark::utility::collect_dist_matrix(base_data_t::context.comm,
            base_data_t::context.rank == 0,
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

} } /** namespace skylark::sketch */

#endif // FJLT_ELEMENTAL_HPP
