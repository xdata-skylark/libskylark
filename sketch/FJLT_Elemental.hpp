#ifndef FJLT_ELEMENTAL_HPP
#define FJLT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "config.h"

#include "context.hpp"
#include "transforms.hpp"

namespace skylark {
namespace sketch {

/**
 * Specialization for [*, SOMETHING]
 */
template <typename ValueType, elem::Distribution ColDist>
struct FJLT_t <
        elem::DistMatrix<ValueType, ColDist, elem::STAR>, /*InputMatrix*/
        elem::Matrix<ValueType> > { /* OutputMatrix */

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;

    typedef elem::DistMatrix<ValueType, elem::STAR, ColDist>
        intermediate_type;
    typedef fft_futs<double>::DCT transform_type;

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
        underlying_transform.apply(inter_A, inter_A,
                                   skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(S, inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)N / (double)S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < S; i++) {
                int row = samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        skylark::utility::collect_dist_matrix(context.comm, context.rank == 0,
            dist_sketch_A, sketch_A);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and rowwise.
     */
    void apply_impl_vdist(const matrix_type& A,
                    output_matrix_type& sketch_of_A,
                    skylark::sketch::rowwise_tag) const {

        // TODO have checked the following code yet.

        // Tranpose to view the input as the underlying transform expects it.
        intermediate_type inter_A(A.Grid());
        elem::Transpose(A, inter_A);

        // Apply the underlying transform
        underlying_transform.apply(inter_A, inter_A,
                                   skylark::sketch::columnwise_tag());

        // Transpose back and keep wanted columns
        matrix_type inter_A_t(A.Grid());
        elem::Transpose(inter_A, inter_A_t);

        // Create the sampled and scaled matrix -- still in distributed mode
        matrix_type dist_sketch_A(A.Height(), S, inter_A.Grid());
        double scale = sqrt((double)N / (double)S);
        for (int j = 0; j < S; j++) {
            int col  = samples[j];
            for (int i = 0; i < A.LocalHeight(); i++) {
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A_t.Matrix().Get(i, col));
            }
        }
    }

    // List of variables associated with this sketch
    /// Input dimension
    const int N;
    /// Output dimension
    const int S;
    const RFUT_t<intermediate_type,
                 transform_type,
                 utility::rademacher_distribution_t<ValueType> >
    /// Underlying mixing (fast-unitary) transform
    underlying_transform;
    std::vector<int> samples;
    /// context for this sketch
    skylark::sketch::context_t& context;

public:

    FJLT_t(int N, int S, skylark::sketch::context_t& context)
        : N(N), S(S), underlying_transform(N, context), samples(S),
          context(context) {

        // The following is sampling with replacement
        boost::random::mt19937 prng(context.newseed());
        boost::random::uniform_int_distribution<int> distribution(0, N - 1);
        for (int i = 0; i < S; i++)
            samples[i] = distribution(prng);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (matrix_type& A,
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
