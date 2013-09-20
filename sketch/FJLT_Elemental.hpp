#ifndef FJLT_ELEMENTAL_HPP
#define FJLT_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "../utility/randgen.hpp"

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
        _underlying_transform.apply(inter_A, inter_A,
                                   skylark::sketch::columnwise_tag());

        // Create the sampled and scaled matrix -- still in distributed mode
        intermediate_type dist_sketch_A(_S, inter_A.Width(), inter_A.Grid());
        double scale = sqrt((double)_N / (double)_S);
        for (int j = 0; j < inter_A.LocalWidth(); j++)
            for (int i = 0; i < _S; i++) {
                int row = _samples[i];
                dist_sketch_A.Matrix().Set(i, j,
                    scale * inter_A.Matrix().Get(row, j));
            }

        skylark::utility::collect_dist_matrix(_context.comm, _context.rank == 0,
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

    // List of variables associated with this sketch
    /// Input dimension
    const int _N;
    /// Output dimension
    const int _S;
    const RFUT_t<intermediate_type,
                 transform_type,
                 utility::rademacher_distribution_t<ValueType> >
    /// Underlying mixing (fast-unitary) transform
    _underlying_transform;
    std::vector<int> _samples;
    /// context for this sketch
    skylark::sketch::context_t& _context;


public:

    FJLT_t(int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _underlying_transform(N, context), _samples(S),
          _context(context) {
        typedef boost::random::uniform_int_distribution<int> distribution_type;
        distribution_type distribution(0, N - 1);

        skylark::utility::random_samples_array_t<value_type, distribution_type>
            random_samples =
            context.allocate_random_samples_array<value_type, distribution_type>
            (S, distribution);
        for (int i = 0; i < S; i++) {
            _samples[i] = random_samples[i];
        }
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
