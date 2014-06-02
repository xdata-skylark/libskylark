#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"

#include "sketch_params.hpp"

namespace skylark { namespace sketch {
/**
 * Specialization distributed input and output in [SOMETHING, *]
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
    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR> matrix_type;
    typedef elem::DistMatrix<value_type, ColDist, elem::STAR>
    output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                   ValueDistribution> data_type;

    /**
     * Regular Constructor
     */
    dense_transform_t (int N, int S, base::context_t& context)
        : data_type (N, S, context) {

    }

    /**
     * Copy constructor
     */
    dense_transform_t (dense_transform_t<matrix_type,
                                         output_matrix_type,
                                         ValueDistribution>& other)
        : data_type(other) {}

    /**
     * Constructor from data
     */
    dense_transform_t(const dense_transform_data_t<value_type,
                                            ValueDistribution>& other_data)
        : data_type(other_data) {}

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

private:

    /**
     * High-performance OPTIMIZED implementation
     */
    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::STAR> R1(grid);
        elem::DistMatrix<value_type, ColDist, elem::STAR>
            A_Top(grid),
            A_Bottom(grid),
            A0(grid),
            A1(grid),
            A2(grid);
        elem::DistMatrix<value_type, ColDist, elem::STAR>
            sketch_of_A_Left(grid),
            sketch_of_A_Right(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid),
            sketch_of_A1_Top(grid),
            sketch_of_A1_Bottom(grid),
            sketch_of_A10(grid),
            sketch_of_A11(grid),
            sketch_of_A12(grid);

        // TODO: are alignments necessary?

        elem::PartitionRight
        ( sketch_of_A,
          sketch_of_A_Left, sketch_of_A_Right, 0 );

        // TODO: Allow for different blocksizes in "down" and "right" directions
        int blocksize = get_blocksize();
        if (blocksize == 0) {
            blocksize = std::min(sketch_of_A.Height(), sketch_of_A.Width());
        }
        int base = 0;
        while (sketch_of_A_Right.Width() > 0) {

            int b = std::min(sketch_of_A_Right.Width(), blocksize);
            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Width());

            elem::RepartitionRight
            ( sketch_of_A_Left, /**/               sketch_of_A_Right,
              sketch_of_A0,     /**/ sketch_of_A1, sketch_of_A2,      b );

            elem::LockedPartitionDown
            ( A, A_Top,
                 A_Bottom, 0 );

            elem::PartitionDown
            ( sketch_of_A1, sketch_of_A1_Top,
                            sketch_of_A1_Bottom, 0 );

            while(A_Bottom.Height() > 0) {
                elem::LockedRepartitionDown
                ( A_Top,    A0,
                  /**/      /**/
                            A1,
                  A_Bottom, A2, b );

                elem::RepartitionDown
                ( sketch_of_A1_Top,    sketch_of_A10,
                  /**/                 /**/
                                       sketch_of_A11,
                  sketch_of_A1_Bottom, sketch_of_A12, b );

                // Local Gemm
                base::Gemm(elem::NORMAL,
                           elem::TRANSPOSE,
                           value_type(1),
                           A1.LockedMatrix(),
                           R1.LockedMatrix(),
                           sketch_of_A11.Matrix());

                elem::SlideLockedPartitionDown
                ( A_Top,    A0,
                            A1,
                  /**/      /**/
                  A_Bottom, A2 );

                elem::SlidePartitionDown
                ( sketch_of_A1_Top,    sketch_of_A10,
                                       sketch_of_A11,
                  /**/                 /**/
                  sketch_of_A1_Bottom, sketch_of_A12 );
            }

            base = base + b;

            elem::SlidePartitionRight
            ( sketch_of_A_Left,               /**/ sketch_of_A_Right,
              sketch_of_A0,     sketch_of_A1, /**/ sketch_of_A2 );
        }
    }


    /**
     * High-performance implementation
     */

    // Communication demanding scenario: Memory-oblivious mode
    // TODO: Block-by-block mode
    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, ColDist> R(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            sketch_of_A_STAR_STAR(grid);

        data_type::realize_matrix_view(R);

        // TODO: is alignment necessary?

        // Global size of the result of the Local Gemm that follows
        sketch_of_A_STAR_STAR.Resize(R.Height(),
                                     A.Width());

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::NORMAL,
                   value_type(1),
                   R.LockedMatrix(),
                   A.LockedMatrix(),
                   sketch_of_A_STAR_STAR.Matrix());

        // Reduce-scatter within process grid
        sketch_of_A.SumScatterFrom(sketch_of_A_STAR_STAR);

    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::STAR> R1(grid);
        elem::DistMatrix<value_type, ColDist, elem::STAR>
            A_Left(grid),
            A_Right(grid),
            A0(grid),
            A1(grid),
            A2(grid);

        // Zero sketch_of_A
        elem::Zero(sketch_of_A);

        // TODO: is alignment necessary?
        R1.AlignWith(sketch_of_A);

        elem::LockedPartitionRight
        ( A,
          A_Left, A_Right, 0 );

        int blocksize = get_blocksize();
        if (blocksize == 0) {
            blocksize = A_Right.Width();
        }
        int base = 0;
        while (A_Right.Width() > 0) {

            int b = std::min(A_Right.Width(), blocksize);
            data_type::realize_matrix_view(R1, 0,                   base,
                                               sketch_of_A.Width(), b);

            elem::RepartitionRight
            ( A_Left, /**/     A_Right,
              A0,     /**/ A1, A2,      b );

            // Local Gemm
            base::Gemm(elem::NORMAL,
                       elem::TRANSPOSE,
                       value_type(1),
                       A1.LockedMatrix(),
                       R1.LockedMatrix(),
                       value_type(1),
                       sketch_of_A.Matrix());

            base = base + b;

            elem::SlidePartitionRight
            ( A_Left,     /**/ A_Right,
              A0,     A1, /**/ A2 );

        }
    }


    /**
     * High-performance implementation
     */

    // Communication demanding scenario: Memory-oblivious mode
    // TODO: Block-by-block mode
    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, ColDist, elem::STAR> R(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            A_STAR_STAR(grid);

        // TODO: are alignments necessary?
        R.AlignWith(sketch_of_A);
        A_STAR_STAR.AlignWith(sketch_of_A);

        data_type::realize_matrix_view(R);

        // Zero sketch_of_A
        elem::Zero(sketch_of_A);

        // Allgather within process grid
        A_STAR_STAR = A;

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::NORMAL,
                   value_type(1),
                   R.LockedMatrix(),
                   A_STAR_STAR.LockedMatrix(),
                   value_type(1),
                   sketch_of_A.Matrix());
    }


    /**
     * High-performance OPTIMIZED implementation
     */
    void matrix_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::STAR> R1(grid);
        elem::DistMatrix<value_type, ColDist, elem::STAR>
            sketch_of_A_Left(grid),
            sketch_of_A_Right(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid);

        // TODO: is alignment necessary?
        R1.AlignWith(sketch_of_A);

        elem::PartitionRight
        ( sketch_of_A,
          sketch_of_A_Left, sketch_of_A_Right, 0 );

        int blocksize = get_blocksize();
        if (blocksize == 0) {
            blocksize = sketch_of_A_Right.Width();
        }
        int base = 0;
        while (sketch_of_A_Right.Width() > 0) {

            int b = std::min(sketch_of_A_Right.Width(), blocksize);
            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Width());

            elem::RepartitionRight
            ( sketch_of_A_Left, /**/               sketch_of_A_Right,
              sketch_of_A0,     /**/ sketch_of_A1, sketch_of_A2,      b );

            // Local Gemm
            base::Gemm(elem::NORMAL,
                       elem::TRANSPOSE,
                       value_type(1),
                       A.LockedMatrix(),
                       R1.LockedMatrix(),
                       sketch_of_A1.Matrix());

            base = base + b;

            elem::SlidePartitionRight
            ( sketch_of_A_Left,               /**/ sketch_of_A_Right,
              sketch_of_A0,     sketch_of_A1, /**/ sketch_of_A2 );

        }
    }


    /**
     * High-performance implementation
     */

    // Communication demanding scenario: Memory-oblivious mode
    // TODO: Block-by-block mode
    void panel_matrix_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, ColDist> R(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            sketch_of_A_STAR_STAR(grid);

        // TODO: is alignment necessary?
        sketch_of_A_STAR_STAR.AlignWith(A);

        data_type::realize_matrix_view(R);

        // Global size of the result of the Local Gemm that follows
        sketch_of_A_STAR_STAR.Resize(R.Height(),
                                     A.Width());

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::NORMAL,
                   value_type(1),
                   R.LockedMatrix(),
                   A.LockedMatrix(),
                   sketch_of_A_STAR_STAR.Matrix());

        // Reduce-scatter within process grid
        sketch_of_A.SumScatterFrom(sketch_of_A_STAR_STAR);
    }


    void sketch_gemm(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        const int sketch_height = sketch_of_A.Height();
        const int sketch_width  = sketch_of_A.Width();
        const int width         = A.Width();

        const double factor = get_factor();

        if((sketch_height * factor <= width) &&
            (sketch_width * factor <= width)) {
            inner_panel_gemm(A, sketch_of_A, tag);
        } else if((sketch_height >= width * factor) &&
            (sketch_width >= width * factor))
            outer_panel_gemm(A, sketch_of_A, tag);
        else
            matrix_panel_gemm(A, sketch_of_A, tag);
    }


    void sketch_gemm(const matrix_type& A,
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


    void apply_impl_vdist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag tag) const {

        sketch_gemm(A, sketch_of_A, tag);
    }


    void apply_impl_vdist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag tag) const {

        sketch_gemm(A, sketch_of_A, tag);
    }
};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_HPP
