#ifndef SPARSET_ELEMENTAL_HPP
#define SPARSET_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"

#include "boost/random/cauchy_distribution.hpp"


namespace skylark {
namespace sketch {


template <typename ValueType,
          elem::Distribution ColDist,
          typename IdxDistributionType,
          template <typename> class ValueDistributionType>
struct hash_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType>,
    IdxDistributionType,
    ValueDistributionType > {

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    typedef IdxDistributionType idx_distribution_type;
    typedef ValueDistributionType<value_type> value_distribution_type;

    /**
     * Constructor
     * Create an object with a particular seed value.
     */
    hash_transform_t (int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _context(context) {

        _row_idx.resize(N);
        _row_value.resize(N);

        boost::random::mt19937 prng(context.newseed());
        idx_distribution_type   row(0, _S - 1);
        value_distribution_type row_value;

        for (int i = 0; i < N; ++i) {
            _row_idx[i]   = row(prng);
            _row_value[i] = row_value(prng);
        }
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A, output_matrix_type& sketch_of_A,
                Dimension dimension) {

        switch(ColDist) {
        case elem::VR:
        case elem::VC:
            apply_impl_vdist (A, sketch_of_A, dimension);
            break;

        default:
            std::cerr << "Unsupported for now..." << std::endl;
            break;
        }
    }


private:

    /// Input dimension
    const int _N;
    /// Output dimension
    const int _S;
    /// context for this sketch
    skylark::sketch::context_t& _context;

protected:
    // Precomputed row index and value per column of Pi
    // All implementations of hash transforms should share this.
    // (to allow copy/change type and modification in derived classes)
    std::vector<int> _row_idx;
    std::vector<value_type> _row_value;

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the column-wise direction of sketching.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) {

        // Create space to hold local part of SA
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
                                          sketch_of_A.Width(),
                                          sketch_of_A.LDim());

        //XXX: newly created matrix is not zeroed!
        elem::Zero(SA_part);

        // Construct Pi * A (directly on the fly)
        for (size_t j = 0; j < A.LocalHeight(); j++) {

            size_t col_idx = A.ColShift() + A.ColStride() * j;

            size_t row_idx          = _row_idx[col_idx];
            value_type scale_factor = _row_value[col_idx];

            for(size_t i = 0; i < A.LocalWidth(); i++) {
                value_type value = scale_factor * A.GetLocal(j, i);
                SA_part.Update(row_idx, A.RowShift() + i, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::reduce (_context.comm,
                            SA_part.LockedBuffer(),
                            SA_part.MemorySize(),
                            sketch_of_A.Buffer(),
                            std::plus<value_type>(),
                            0);
    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for the row-wise direction of sketching.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::rowwise_tag) {

        // Create space to hold local part of SA
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
                                          sketch_of_A.Width(),
                                          sketch_of_A.LDim());

        elem::Zero(SA_part);

        // Construct A * Pi (directly on the fly)
        for (size_t j = 0; j < A.LocalHeight(); ++j) {

            size_t row_idx = A.ColShift() + A.ColStride() * j;

            for(size_t i = 0; i < A.LocalWidth(); ++i) {

                size_t col_idx     = A.RowShift() + A.RowStride() * i;
                size_t new_col_idx = _row_idx[col_idx];
                value_type value   = _row_value[col_idx] * A.GetLocal(j, i);

                SA_part.Update(row_idx, new_col_idx, value);
            }
        }

        // Pull everything to rank-0
        boost::mpi::reduce (_context.comm,
                            SA_part.LockedBuffer(),
                            SA_part.MemorySize(),
                            sketch_of_A.Buffer(),
                            std::plus<value_type>(),
                            0);
    }
};

} // namespace sketch
} // namespace skylark

#endif // SPARSET_ELEMENTAL_HPP
