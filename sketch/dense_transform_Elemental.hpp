#ifndef DENSE_TRANSFORM_ELEMENTAL_HPP
#define DENSE_TRANSFORM_ELEMENTAL_HPP

#include <elemental.hpp>

#include "context.hpp"
#include "dense_transform_data.hpp"
#include "transforms.hpp"
#include "../utility/comm.hpp"
#include "../utility/exception.hpp"
#include "../utility/randgen.hpp"


namespace skylark { namespace sketch {

/**
 * Specialization distributed input, local output, for [*, SOMETHING]
 */
template <typename ValueType,
          elem::Distribution ColDist,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::Matrix<ValueType>,
    ValueDistribution > :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::Matrix<value_type> output_matrix_type;
    // Typedef distribution
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                  ValueDistribution> base_data_t;

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


        elem::Matrix<value_type> S_local(base_data_t::S, slice_width);
        for (int js = 0; js < A.LocalHeight(); js += slice_width) {
            int je = std::min(js + slice_width, A.LocalHeight());
            // adapt size of local portion (can be less than slice_width)
            S_local.ResizeTo(base_data_t::S, je-js);
            for(int j = js; j < je; j++) {
                int col = A.RowShift() + A.RowStride() * j;
                for (int i = 0; i < base_data_t::S; i++) {
                    value_type sample =
                        base_data_t::random_samples[col * base_data_t::S + i];
                    S_local.Set(i, j-js, base_data_t::scale * sample);
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
        boost::mpi::reduce (base_data_t::context.comm,
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
        matrix_type SA_dist(A.Height(), base_data_t::S, A.Grid());

        // Create S. Since it is rowwise, we assume it can be held in memory.
        elem::Matrix<value_type> S_local(base_data_t::S, base_data_t::N);
        for (int j = 0; j < base_data_t::N; j++) {
            for (int i = 0; i < base_data_t::S; i++) {
                value_type sample =
                    base_data_t::random_samples[j * base_data_t::S + i];
                S_local.Set(i, j, base_data_t::scale * sample);
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
        skylark::utility::collect_dist_matrix(base_data_t::context.comm,
            base_data_t::context.rank == 0,
            SA_dist, sketch_of_A);
    }

public:
    /**
     * Constructor
     * Create an object with a particular seed value.
     */
    dense_transform_t (int N, int S, skylark::sketch::context_t& context)
        : base_data_t (N, S, context) {}

    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    dense_transform_t(const dense_transform_data_t<value_type,
        ValueDistribution>& other_data) :
        base_data_t(other_data.get_data()) {}

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
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    elem::DistMatrix<ValueType, ColDist, elem::STAR>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {
    // Typedef matrix type so that we can use it regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR>
    output_matrix_type;
    // Typedef distribution
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                  ValueDistribution> base_data_t;

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
        elem::Matrix<value_type> S_local(base_data_t::S, base_data_t::N);
        for (int j = 0; j < base_data_t::N; j++) {
            for (int i = 0; i < base_data_t::S; i++) {
                value_type sample =
                    base_data_t::random_samples[j * base_data_t::S + i];
                S_local.Set(i, j, base_data_t::scale * sample);
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

public:
    /**
     * Constructor
     * Create an object with a particular seed value.
     */
    dense_transform_t (int N, int S, skylark::sketch::context_t& context)
        : base_data_t (N, S, context) {}

    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValueDistribution>& other) :
        base_data_t(other.get_data()) {}

    dense_transform_t(const dense_transform_data_t<value_type,
        ValueDistribution>& other_data) :
        base_data_t(other_data.get_data()) {}

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

#endif // DENSE_TRANSFORM_ELEMENTAL_HPP
