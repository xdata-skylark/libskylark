#ifndef FJLT_ELEMENTAL_HPP
#define FJLT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "FJLT_data.hpp"
#include "transforms.hpp"
#include "../utility/randgen.hpp"

namespace skylark { namespace sketch {
/**
 * Specialization for [*, SOMETHING]
 */
template <typename ValueType, elem::Distribution ColDist>
struct FJLT_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>, /*InputMatrix*/
    elem::Matrix<ValueType> > :
        public FJLT_data_t<ValueType>
{ /* OutputMatrix */

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef elem::DistMatrix<ValueType, elem::STAR, ColDist>
    intermediate_type;
    typedef fft_futs<double>::DCT transform_type;
    typedef FJLT_data_t<value_type> base_data_t;
    typedef utility::rademacher_distribution_t<value_type>
    underlying_value_distribution_type;
    typedef RFUT_t<intermediate_type,
                   transform_type,
                   underlying_value_distribution_type>
    underlying_type;

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


public:

    FJLT_t(int N, int S, skylark::sketch::context_t& context)
        : base_data_t (N, S, context) {}

    FJLT_t(FJLT_t<matrix_type,
                  output_matrix_type>& other)
    : base_data_t(other.get_data()) {}


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
            apply_impl_vdist (A, sketch_of_A, dimension);
            break;

        default:
            std::cerr << "Unsupported for now..." << std::endl;
            break;
        }
    }
};

} // namespace sketch
} // namespace skylark

#endif // FJLT_ELEMENTAL_HPP
