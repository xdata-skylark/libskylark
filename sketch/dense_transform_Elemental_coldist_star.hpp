#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_HPP

#include "../base/base.hpp"

#include "transforms.hpp"
#include "dense_transform_data.hpp"
#include "../utility/comm.hpp"
#include "../utility/get_communicator.hpp"

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL
#include "sketch_params.hpp"
#endif

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

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL

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

        int base = 0;

        elem::PartitionRight
        ( sketch_of_A,
          sketch_of_A_Left, sketch_of_A_Right, 0 );

        // TODO: Should it be sketch_of_A_Right.Width() instead?
        // TODO: Allow for different blocksizes in Down and Right partitionings
        while (sketch_of_A_Left.Width() > 0) {
            int b = get_blocksize();

            base_data_t::realize_matrix_view(R1, 0, base, A.Width(), b);

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

                elem::LocalGemm(elem::NORMAL,
                                elem::NORMAL,
                                value_type(1),
                                A1,
                                R1,
                                sketch_of_A11);

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

            // TODO: "added b" should be the min(b, dimension - base)?
            base = base + b;

            elem::SlidePartitionRight
            ( sketch_of_A_Left,               /**/ sketch_of_A_Right,
              sketch_of_A0,     sketch_of_A1, /**/ sketch_of_A2 );
        }
    }


    // Communication demanding scenario
    // TODO: Adapt from [MC, MR]
    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::VC> R1(grid);
        elem::DistMatrix<value_type>
            A_Left(grid),
            A_Right(grid),
            A0(grid),
            A1(grid),
            A2(grid);
        elem::DistMatrix<value_type>
            sketch_of_A_Top(grid),
            sketch_of_A_Bottom(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid),
            sketch_of_A1_Left(grid),
            sketch_of_A1_Right(grid),
            sketch_of_A10(grid),
            sketch_of_A11(grid),
            sketch_of_A12(grid);
        elem::DistMatrix<value_type, elem::VC, elem::STAR>
            A1_VC_STAR(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            sketch_of_A11_STAR_STAR(grid);

        int base = 0;

        elem::PartitionDown
        ( sketch_of_A,
          sketch_of_A_Top, sketch_of_A_Bottom, 0 );

        // TODO: Allow for different blocksizes in Down and Right partitionings
        while (sketch_of_A_Bottom.Height() > 0) {
            int b = get_blocksize();

            // TODO: should it be A.Height() instead of A.Width()?
            base_data_t::realize_matrix_view(R1, base, 0, b, A.Width());

            elem::RepartitionDown
            ( sketch_of_A_Top,     sketch_of_A0,
              /**/                 /**/
                                   sketch_of_A1,
              sketch_of_A_Bottom, sketch_of_A2,  b );


            // TODO: is alignment necessary?
            A1_VC_STAR.AlignWith(R1);

            elem::LockedPartitionRight
            ( A,
              A_Left, A_Right, 0 );

            elem::PartitionRight
            ( sketch_of_A1,
              sketch_of_A1_Left, sketch_of_A1_Right, 0 );

            while(A_Right.Width() > 0) {

                elem::LockedRepartitionRight
                ( A_Left, /**/     A_Right,
                  A0,     /**/ A1, A2,      b );

                elem::RepartitionRight
                ( sketch_of_A1_Left, /**/                sketch_of_A1_Right,
                  sketch_of_A10,     /**/ sketch_of_A11, sketch_of_A12,      b);


                // [MC, MR] -> [VC, STAR]:
                // TODO: Describe the communication pattern
                A1_VC_STAR = A1;

                elem::LocalGemm(elem::NORMAL,
                                elem::NORMAL,
                                value_type(1),
                                R1,
                                A1_VC_STAR,
                                sketch_of_A11_STAR_STAR);

                // [MC, MR].sum-scatter-update([STAR, STAR]):
                // - sum all [STAR, STAR] matrices over the entire process grid
                // - scatter to form [MC, MR]
                // DONE: Should it be SumScatterFrom() in Elemental-0.81?
                // DONE: Replace with Elemental-0.83 equivalent
                sketch_of_A11.SumScatterUpdate(value_type(1),
                    sketch_of_A11_STAR_STAR);

                elem::SlideLockedPartitionRight
                ( A_Left,     /**/ A_Right,
                  A0,     A1, /**/ A2 );

                elem::SlidePartitionRight
                ( sketch_of_A1_Left,                /**/ sketch_of_A1_Right,
                  sketch_of_A10,     sketch_of_A11, /**/ sketch_of_A12 );

            }

            // TODO: "added b" should be the min(b, dimension - base)?
            base = base + b;

            elem::SlidePartitionDown
            ( sketch_of_A_Top,    sketch_of_A0,
                                  sketch_of_A1,
              /**/                /**/
              sketch_of_A_Bottom, sketch_of_A2 );

        }
    }


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

        // TODO: is alignment necessary?
        R1.AlignWith(sketch_of_A);

        int base = 0;
        elem::LockedPartitionRight
        ( A,
          A_Left, A_Right, 0 );

        while (A_Right.Width() > 0) {
            int b = get_blocksize();

            // TODO: Should the order of arguments be:
            // (R1, base, 0, b, sketch_of_A.Width())?
            base_data_t::realize_matrix_view(R1, 0, base, A.Height(), b);

            elem::RepartitionRight
            ( A_Left, /**/     A_Right,
              A0,     /**/ A1, A2,      b );

            elem::LocalGemm(elem::NORMAL,
                            elem::NORMAL,
                            value_type(1),
                            A1,
                            R1,
                            value_type(1),
                            sketch_of_A);

            // TODO: "added b" should be the min(b, dimension - base)?
            base = base + b;

            elem::SlidePartitionRight
            ( A_Left,     /**/ A_Right,
              A0,     A1, /**/ A2 );

        }
    }


    // Communication demanding scenario
    // TODO: Adapt from [MC, MR]
    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::MC, elem::STAR> R1(grid);
        elem::DistMatrix<value_type>
            A_Top(grid),
            A_Bottom(grid),
            A0(grid),
            A1(grid),
            A2(grid);
        elem::DistMatrix<value_type, elem::MR, elem::STAR>
            A1Trans_MR_STAR(grid);

        // TODO: are alignments necessary?
        R1.AlignWith(sketch_of_A);
        A1Trans_MR_STAR.AlignWith(sketch_of_A);

        int base = 0;

        elem::LockedPartitionDown
        ( A,
          A_Top, A_Bottom, 0 );

        while (A_Bottom.Height() > 0) {
            int b = get_blocksize();

            // TODO: should it be sketch_of_A.Height() instead of A.Height()?
            base_data_t::realize_matrix_view(R1, 0, base, A.Height(), b);

            elem::RepartitionDown
             ( A_Top,    A0,
               /**/      /**/
                         A1,
               A_Bottom, A2, b );


            // [MC, MR].transpose-col-all-gather([MR, STAR]):
            // TODO: Describe the communication pattern
            // TODO: Describe cache benefits from transposition:
            //       why not simply use A1[STAR, MR]?
            // DONE: Replace with Elemental-0.83 equivalent
            // A1Trans_MR_STAR.TransposeFrom(A1);
            A1.TransposeColAllGather(A1Trans_MR_STAR);

            elem::LocalGemm(elem::NORMAL,
                            elem::TRANSPOSE,
                            value_type(1),
                            R1,
                            A1Trans_MR_STAR,
                            value_type(1),
                            sketch_of_A);

            // TODO: "added b" should be the min(b, dimension - base)?
            base = base + b;

            elem::SlidePartitionDown
            ( A_Top,    A0,
                        A1,
              /**/      /**/
              A_Bottom, A2 );
        }
    }


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

        int base = 0;

        elem::LockedPartitionRight
        ( sketch_of_A,
          sketch_of_A_Left, sketch_of_A_Right, 0 );

        while (sketch_of_A_Right.Width() > 0) {
            int b = get_blocksize();

            // TODO: Should it be A.Width() instead of A.Height()?
            base_data_t::realize_matrix_view(R1, 0, base, A.Height(), b);

            elem::RepartitionRight
            ( sketch_of_A_Left, /**/               sketch_of_A_Right,
              sketch_of_A0,     /**/ sketch_of_A1, sketch_of_A2,      b );

            elem::LocalGemm(elem::NORMAL,
                            elem::NORMAL,
                            value_type(1),
                            A,
                            R1,
                            sketch_of_A1);

            // TODO: "added b" should be the min(b, dimension - base)?
            base = base + b;

            elem::SlidePartitionRight
            ( sketch_of_A_Left,               /**/ sketch_of_A_Right,
              sketch_of_A0,     sketch_of_A1, /**/ sketch_of_A2 );

        }
    }


    // Communication demanding scenario
    // TODO: Adapt from [MC, MR]
    void panel_matrix_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        // TODO: Should it be R1[STAR, MC] instead of R1[STAR, VC]?
        elem::DistMatrix<value_type, elem::STAR, elem::VC> R1(grid);

        elem::DistMatrix<value_type>
            sketch_of_A_Top(grid),
            sketch_of_A_Bottom(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid);
        elem::DistMatrix<value_type, elem::MR, elem::STAR>
            sketch_of_A_temp(grid);

        // TODO: is alignment necessary?
        sketch_of_A_temp.AlignWith(A);

        int base = 0;
        elem::PartitionDown
        ( sketch_of_A,
          sketch_of_A_Top, sketch_of_A_Bottom, 0 );

        while (sketch_of_A_Bottom.Height() > 0) {
            int b = get_blocksize();

            // TODO: should it be A.Height() instead of A.Width()?
            base_data_t::realize_matrix_view(R1, base, 0, b, A.Width());

            elem::RepartitionDown
            ( sketch_of_A_Top,     sketch_of_A0,
              /**/                 /**/
                                   sketch_of_A1,
              sketch_of_A_Bottom, sketch_of_A2, b );


            // A.T[MR, MC] * R1.T[MC, STAR] = (A.T * R1.T)[MR, STAR]:
            // extra summation needed because MC index set has gaps
            elem::LocalGemm(elem::TRANSPOSE,
                            elem::TRANSPOSE,
                            value_type(1),
                            A,
                            R1,
                            sketch_of_A_temp);

            // TODO: Describe cache benefits from transposition of terms
            //       and implicit transposition of result after the summation
            // TODO: Describe the effect of
            //       [MC, MR].transpose-sum-scatter-update([MR, STAR]):
            //       sum over column communicators and keep subset of values?
            // DONE: Check/substitute proper Elemental-0.83 equivalent
            sketch_of_A1.TransposeColSumScatterUpdate(value_type(1),
                sketch_of_A_temp);

            // TODO: "added b" should be the min(b, dimension - base)?
            base = base + b;

            elem::SlidePartitionDown
            ( sketch_of_A_Top,    sketch_of_A0,
                                  sketch_of_A1,
              /**/                /**/
              sketch_of_A_Bottom, sketch_of_A2 );
        }
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


    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag tag) const {

        sketch_gemm(A, sketch_of_A, tag);
    }

#else

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for [VR/VC, *] and columnwise.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& sketch_of_A,
                           columnwise_tag) const {

        // Redistribute matrix A: [VC/VR, STAR] -> [STAR, VC/VR]
        elem::DistMatrix<value_type, elem::STAR, ColDist> A_STAR_ColDist(A);

        elem::DistMatrix<value_type,
                         elem::STAR, ColDist>
            sketch_of_A_STAR_ColDist(sketch_of_A.Height(), sketch_of_A.Width());
        elem::Zero(sketch_of_A_STAR_ColDist);

        // Matrix S carries the random samples in the sketching operation S*A.
        // We realize S in parts and compute in a number of local rounds.
        // This ensures handling of cases with a huge S.

        // Max memory assigned to S_part at each round (100 MB by default)
        // TODO: Can we optimize this const for the GEMM that follows?
        const int S_PART_MAX_MEMORY = 100000000;

        int S_height = data_type::_S;
        int S_width = data_type::_N;
        int S_row_num_bytes = S_width * sizeof(value_type);

        // TODO: Guard against the case of S_PART_MAX_MEMORY  < S_row_num_bytes
        int S_part_num_rows = S_PART_MAX_MEMORY / S_row_num_bytes;
        int S_num_rows_consumed = 0;

        while (S_num_rows_consumed < S_height) {
            // Setup S_part S which consists of successive rows in S
            int S_part_height = std::min(S_part_num_rows,
                S_height - S_num_rows_consumed);
            elem::Matrix<value_type> S_part(S_part_height,
                S_width);
            elem::Zero(S_part);
            // Fill S_part with appropriate random samples
            for (int i_loc = 0; i_loc < S_part_height; ++i_loc) {
                int i = S_num_rows_consumed + i_loc;
                for(int j = 0; j < S_width; ++j) {
                    value_type sample =
                        data_type::random_samples[j * data_type::_S + i];
                    S_part.Set(i_loc, j, data_type::scale * sample);
                }
            }
            // Setup a view in sketch_of_A to land the result of S_part*A
            elem::Matrix<value_type> sketch_slice;
            elem::View(sketch_slice, sketch_of_A_STAR_ColDist.Matrix(),
                S_num_rows_consumed, 0,
                S_part_height, A_STAR_ColDist.LocalWidth());
            // Do the multiplication: S_part*A
            base::Gemm (elem::NORMAL,
                elem::NORMAL,
                1.0,
                S_part,
                A_STAR_ColDist.LockedMatrix(),
                0.0,
                sketch_slice);
            S_num_rows_consumed += S_part_height;
        }
        // Redistribute the sketch: [STAR, VC/VR] -> [VC/VR, STAR]
        sketch_of_A = sketch_of_A_STAR_ColDist;
    }

    /**
      * Apply the sketching transform that is described in by the sketch_of_A.
      * Implementation for [VR/VC, *] and rowwise.
      */
    void apply_impl_vdist(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          rowwise_tag) const {


        // Create S. Since it is rowwise, we assume it can be held in memory.
        elem::Matrix<value_type> S_local(data_type::_S, data_type::_N);
        for (int j = 0; j < data_type::_N; j++) {
            for (int i = 0; i < data_type::_S; i++) {
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S_local.Set(i, j, data_type::scale * sample);
            }
        }

        // Apply S to the local part of A to get the local part of sketch_of_A.
        base::Gemm(elem::NORMAL,
            elem::TRANSPOSE,
            1.0,
            A.LockedMatrix(),
            S_local,
            0.0,
            sketch_of_A.Matrix());
    }

#endif

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_COLDIST_STAR_HPP
