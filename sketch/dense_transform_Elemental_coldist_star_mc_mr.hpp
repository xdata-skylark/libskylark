#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_MC_MR_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_MC_MR_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"

#include "sketch_params.hpp"

namespace skylark { namespace sketch {
/**
 * Specialization distributed input [SOMETHING, *], distributed output [MC, MR]
 */
template <typename ValueType, El::Distribution ColDist,
          typename ValueAccessor>
struct dense_transform_t <
    El::DistMatrix<ValueType, ColDist, El::STAR>,
    El::DistMatrix<ValueType>,
    ValueAccessor> :
        public dense_transform_data_t<ValueAccessor> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef El::DistMatrix<value_type, ColDist, El::STAR> matrix_type;
    typedef El::DistMatrix<value_type> output_matrix_type;
    typedef dense_transform_data_t<ValueAccessor> data_type;


    /**
     * Regular constructor
     */
    dense_transform_t (int N, int S, double scale, base::context_t& context)
        : data_type (N, S, scale, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValueAccessor>& other)
        : data_type(other) {

    }

    /**
     * Constructor from data
     */
    dense_transform_t(const data_type& other_data)
        : data_type(other_data) {

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
                output_matrix_type& sketch_of_A,
                Dimension dimension) const {

        switch(ColDist) {
        case El::VR:
        case El::VC:
            try {
                apply_impl_dist(A, sketch_of_A, dimension);
            } catch (std::logic_error e) {
                SKYLARK_THROW_EXCEPTION (
                    base::elemental_exception()
                         << base::error_msg(e.what()) );
            } catch(boost::mpi::exception e) {
                SKYLARK_THROW_EXCEPTION (
                    base::mpi_exception()
                        << base::error_msg(e.what()) );
            }
            break;
        default:
            SKYLARK_THROW_EXCEPTION (
                base::unsupported_matrix_distribution() );
        }
    }

    int get_N() const { return this->_N; } /**< Get input dimension. */
    int get_S() const { return this->_S; } /**< Get output dimension. */

    const sketch_transform_data_t* get_data() const { return this; }

private:

    /**
     * High-performance OPTIMIZED implementation
     */
    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This type of sketched_gemm has not been implemented yet."));

    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This type of sketched_gemm has not been implemented yet."));

    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {


        const El::Grid& grid = A.Grid();

        El::DistMatrix<value_type, El::MR, El::STAR> R1(grid);
        El::DistMatrix<value_type, ColDist,  El::STAR>
            A_Left(grid),
            A_Right(grid),
            A0(grid),
            A1(grid),
            A2(grid);
        El::DistMatrix<value_type, El::MC, El::STAR>
            A1_MC_STAR(grid);

        // Zero sketch_of_A
        El::Zero(sketch_of_A);

        // TODO: are alignments necessary?
        R1.AlignWith(sketch_of_A);
        A1_MC_STAR.AlignWith(sketch_of_A);

        El::LockedPartitionRight
        ( A,
          A_Left, A_Right, 0 );

        int blocksize = get_blocksize();
        if (blocksize == 0) {
            blocksize = A_Right.Width();
        }
        int base = 0;
        while (A_Right.Width() > 0) {

            int b = std::min(static_cast<int>(A_Right.Width()), blocksize);
            data_type::realize_matrix_view(R1, 0,                   base,
                                               sketch_of_A.Width(), b);

            El::RepartitionRight
            ( A_Left, /**/      A_Right,
              A0,     /**/ A1,  A2,      b );

            // Allgather within process columns
            A1_MC_STAR = A1;

            // Local Gemm
            base::Gemm(El::NORMAL,
                       El::TRANSPOSE,
                       value_type(1),
                       A1_MC_STAR.LockedMatrix(),
                       R1.LockedMatrix(),
                       value_type(1),
                       sketch_of_A.Matrix());

            base = base + b;

            El::SlidePartitionRight
            ( A_Left,     /**/ A_Right,
              A0,     A1, /**/ A2 );

        }
    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This type of sketched_gemm has not been implemented yet."));

    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void matrix_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This type of sketched_gemm has not been implemented yet."));

    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void panel_matrix_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        SKYLARK_THROW_EXCEPTION (
          base::sketch_exception()
              << base::error_msg(
                 "This type of sketched_gemm has not been implemented yet."));

    }


    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag tag) const {

        const int sketch_height = sketch_of_A.Height();
        const int sketch_width  = sketch_of_A.Width();
        const int width         = A.Width();

        const double factor = get_factor();

        if((sketch_height * factor <= width) &&
            (sketch_width * factor <= width))
            inner_panel_gemm(A, sketch_of_A, tag);
        else if((sketch_height >= width * factor) &&
            (sketch_width >= width * factor))
            outer_panel_gemm(A, sketch_of_A, tag);
        else
            matrix_panel_gemm(A, sketch_of_A, tag);
    }


    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag tag) const {

        const int sketch_height = sketch_of_A.Height();
        const int sketch_width  = sketch_of_A.Width();
        const int height         = A.Height();

        const double factor = get_factor();

        if((sketch_height * factor <= height) &&
            (sketch_width * factor <= height))
            inner_panel_gemm(A, sketch_of_A, tag);
        else if((sketch_height >= height * factor) &&
            (sketch_width >= height * factor))
            outer_panel_gemm(A, sketch_of_A, tag);
        else
            panel_matrix_gemm(A, sketch_of_A, tag);
    }

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_MC_MR_HPP
