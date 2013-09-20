#ifndef DENSET_ELEMENTAL_HPP
#define DENSET_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "transforms.hpp"
#include "../utility/comm.hpp"
#include "../utility/exception.hpp"
#include "../utility/randgen.hpp"


namespace skylark {
namespace sketch {

/**
 * Specialization distributed input, local output, for [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class DistributionType>
struct dense_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType>,
    DistributionType> {

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    // Typedef distribution
    typedef DistributionType<value_type> distribution_type;
private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) const {

        // Create space to hold partial SA --- for 1D, we need SA space
        elem::Matrix<value_type> SA_part (sketch_of_A.Height(),
                                          sketch_of_A.Width(),
                                          sketch_of_A.LDim());
        elem::Zero(SA_part);

        // To avoid allocating a huge S_local matrix we are breaking
        // S_local into column slices, and multiply one by one.
        // The number of columns in each slice is A's width
        // since that way the slice take the same amount of memory as
        // the sketch.

        int slice_width = A.Width();


        elem::Matrix<value_type> S_local(_S, slice_width);

        for (int js = 0; js < A.LocalHeight(); js += slice_width) {
            int je = std::min(js + slice_width, A.LocalHeight());
            // adapt size of local portion (can be less than slice_width)
            S_local.ResizeTo(_S, je-js);
            for(int j = js; j < je; j++) {
                int col = A.RowShift() + A.RowStride() * j;
                for (int i = 0; i < _S; i++) {
                    value_type sample = _random_samples[col * _S + i];
                    S_local.Set(i, j-js, scale * sample);
                }
            }

            elem::Matrix<value_type> A_slice;
            elem::LockedView(A_slice, A.LockedMatrix(),
                js, 0, je-js, A.Width());

            // Do the multiplication
            elem::Gemm (elem::NORMAL,
                elem::NORMAL,
                1.0,
                S_local,
                A_slice,
                1.0,
                SA_part);
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
      * Implementation for [VR/VC, *] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        // Create a distributed matrix to hold the output.
        //  We later gather to a dense matrix.
        matrix_type SA_dist(A.Height(), _S, A.Grid());

        // Create S. Since it is rowwise, we assume it can be held in memory.
        elem::Matrix<value_type> S_local(_S, _N);
        for (int j = 0; j < _N; j++) {
            for (int i = 0; i < _S; i++) {
                value_type sample = _random_samples[j * _S + i];
                S_local.Set(i, j, scale * sample);
            }
        }

        // Apply S to the local part of A to get the local part of SA.
        elem::Gemm(elem::NORMAL,
            elem::TRANSPOSE,
            1.0,
            A.LockedMatrix(),
            S_local,
            0.0,
            SA_dist.Matrix());

        // Collect at rank 0.
        // TODO Grid rank 0 or context rank 0?
        skylark::utility::collect_dist_matrix(_context.comm, _context.rank == 0,
            SA_dist, sketch_of_A);
    }

    // List of variables associated with this sketch
    /// Input dimension
    const int _N;
    /// Output dimension
    const int _S;
    /// Context for this sketch
    skylark::sketch::context_t& _context;
    /// Random samples
    skylark::utility::random_samples_array_t<value_type, distribution_type>
     _random_samples;

protected:
    double scale;

public:
    /**
     * Constructor
     * Create an object with a particular seed value.
     */
    dense_transform_t (int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _context(context) {
        distribution_type distribution;
        _random_samples =
            context.allocate_random_samples_array<value_type, distribution_type>
            (N * S, distribution);
        // No scaling in "raw" form
        scale = 1.0;
    }


    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {

        switch(ColDist) {
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
};

/**
 * Specialization distributed input and output in [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class DistributionType>
struct dense_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    DistributionType> {

public:
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> output_matrix_type;
    // Typedef distribution
    typedef DistributionType<value_type> distribution_type;

private:
    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           skylark::sketch::columnwise_tag) const {

        // TODO no point in implementing this now as the implementation
        //      will depend on how the random numbers are generated.
    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [VR/VC, *] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {


        // Create S. Since it is rowwise, we assume it can be held in memory.
        elem::Matrix<value_type> S_local(_S, _N);
        for (int j = 0; j < _N; j++) {
            for (int i = 0; i < _S; i++) {
                value_type sample = _random_samples[j * _S + i];
                S_local.Set(i, j, scale * sample);
            }
        }

        // Apply S to the local part of A to get the local part of sketch_of_A.
        elem::Gemm(elem::NORMAL,
            elem::TRANSPOSE,
            1.0,
            A.LockedMatrix(),
            S_local,
            0.0,
            sketch_of_A.Matrix());
    }

    // List of variables associated with this sketch
    /// Input dimension
    const int _N;
    /// Output dimension
    const int _S;
    /// context for this sketch
    skylark::sketch::context_t& _context;
    /// Random samples
    skylark::utility::random_samples_array_t<value_type, distribution_type>
     _random_samples;

protected:
    double scale;

public:
    /**
     * Constructor
     */
    dense_transform_t (int N, int S, skylark::sketch::context_t& context)
        : _N(N), _S(S), _context(context) {
        distribution_type distribution;
        _random_samples =
            context.allocate_random_samples_array<value_type, distribution_type>
            (N * S, distribution);
        // No scaling in "raw" form
        scale = 1.0;
    }


    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {

        switch(ColDist) {
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
};

} // namespace sketch
} // namespace skylark

#endif // DENSET_ELEMENTAL_HPP
