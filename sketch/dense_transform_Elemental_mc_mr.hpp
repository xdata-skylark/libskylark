#ifndef SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_HPP
#define SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_HPP

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
 * Specialization distributed input [MC, MR], distributed output [MC, MR]
 */
template <typename ValueType,
          template <typename> class ValueDistribution>
struct dense_transform_t <
    elem::DistMatrix<ValueType>,
    elem::DistMatrix<ValueType>,
    ValueDistribution> :
        public dense_transform_data_t<ValueType,
                                      ValueDistribution> {

    // Typedef matrix and distribution types so that we can use them regularly
    typedef ValueType value_type;
    typedef elem::DistMatrix<value_type> matrix_type;
    typedef elem::DistMatrix<value_type> output_matrix_type;
    typedef ValueDistribution<value_type> value_distribution_type;
    typedef dense_transform_data_t<ValueType,
                                   ValueDistribution> data_type;

    /**
     * Regular constructor
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
    }

private:

#ifdef HP_DENSE_TRANSFORM_ELEMENTAL_MC_MR

////////////////////////////////////////////////////////////////////////////////
#ifdef OPTIMIZED // OPTIMIZED

    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::VR> R1(grid);
        elem::DistMatrix<value_type>
            A_Top(grid),
            A_Bottom(grid),
            A0(grid),
            A1(grid),
            A2(grid);
        elem::DistMatrix<value_type>
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
        elem::DistMatrix<value_type, elem::STAR, elem::VR>
            A1_STAR_VR(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            sketch_of_A11_STAR_STAR(grid);

        // TODO: are alignments necessary?

        elem::PartitionRight
        ( sketch_of_A,
          sketch_of_A_Left, sketch_of_A_Right, 0 );

        // TODO: Allow for different blocksizes in "down" and "right" directions
        int blocksize = get_blocksize();
        int base = 0;
        while (sketch_of_A_Right.Width() > 0) {

            int b = std::min(sketch_of_A_Right.Width(), blocksize);
            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Width());

            elem::RepartitionRight
            ( sketch_of_A_Left, /**/               sketch_of_A_Right,
              sketch_of_A0,     /**/ sketch_of_A1, sketch_of_A2,      b );

            // TODO: is alignment necessary?
            A1_STAR_VR.AlignWith(R1);

            elem::LockedPartitionDown
            ( A,
              A_Top, A_Bottom, 0 );

            elem::PartitionDown
            ( sketch_of_A1,
              sketch_of_A1_Top, sketch_of_A1_Bottom, 0 );

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

                // Alltoall within process columns
                A1_STAR_VR = A1;

                // Global size of the result of the Local Gemm that follows
                sketch_of_A11_STAR_STAR.Resize(A1_STAR_VR.Height(),
                                               R1.Height());

                // Local Gemm
                base::Gemm(elem::NORMAL,
                           elem::TRANSPOSE,
                           value_type(1),
                           A1_STAR_VR.LockedMatrix(),
                           R1.LockedMatrix(),
                           sketch_of_A11_STAR_STAR.Matrix());

                // Reduce-scatter within process grid
                sketch_of_A11.SumScatterFrom(sketch_of_A11_STAR_STAR);

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

#else

    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::VR> R(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::VR>
            A_STAR_VR(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            sketch_of_A_STAR_STAR(grid);

        // TODO: are alignments necessary?

        data_type::realize_matrix_view(R);

        // TODO: is alignment necessary?
        A_STAR_VR.AlignWith(R);

        // Alltoall within process columns
        A_STAR_VR = A;

        // Global size of the result of the Local Gemm that follows
        sketch_of_A_STAR_STAR.Resize(A_STAR_VR.Height(),
                    R.Height());

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::TRANSPOSE,
                   value_type(1),
                   A_STAR_VR.LockedMatrix(),
                   R.LockedMatrix(),
                   sketch_of_A_STAR_STAR.Matrix());

        // Reduce-scatter within process grid
        sketch_of_A.SumScatterFrom(sketch_of_A_STAR_STAR);

    }


#endif // OPTIMIZED


////////////////////////////////////////////////////////////////////////////////
#ifdef OPTIMIZED // OPTIMIZED

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

        // TODO: are alignments necessary?

        elem::PartitionDown
        ( sketch_of_A,
          sketch_of_A_Top, sketch_of_A_Bottom, 0 );

        // TODO: Allow for different blocksizes in "down" and "right" directions
        int blocksize = get_blocksize();
        int base = 0;
        while (sketch_of_A_Bottom.Height() > 0) {

            int b = std::min(sketch_of_A_Bottom.Height(), blocksize);
            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Height());

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

                // Alltoall within process rows
                A1_VC_STAR = A1;

                // Global size of the result of the Local Gemm that follows
                sketch_of_A11_STAR_STAR.Resize(R1.Height(),
                                               A1_VC_STAR.Width());

                // Local Gemm
                base::Gemm(elem::NORMAL,
                           elem::NORMAL,
                           value_type(1),
                           R1.LockedMatrix(),
                           A1_VC_STAR.LockedMatrix(),
                           value_type(0),
                           sketch_of_A11_STAR_STAR.Matrix());

                // Reduce-scatter within process grid
                sketch_of_A11.SumScatterFrom(sketch_of_A11_STAR_STAR);

                elem::SlideLockedPartitionRight
                ( A_Left,     /**/ A_Right,
                  A0,     A1, /**/ A2 );

                elem::SlidePartitionRight
                ( sketch_of_A1_Left,                /**/ sketch_of_A1_Right,
                  sketch_of_A10,     sketch_of_A11, /**/ sketch_of_A12 );

            }

            base = base + b;

            elem::SlidePartitionDown
            ( sketch_of_A_Top,    sketch_of_A0,
                                  sketch_of_A1,
              /**/                /**/
              sketch_of_A_Bottom, sketch_of_A2 );

        }
    }

#else

    void inner_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::VC> R(grid);
        elem::DistMatrix<value_type, elem::VC, elem::STAR>
            A_VC_STAR(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::STAR>
            sketch_of_A_STAR_STAR(grid);

        // TODO: are alignments necessary?

        data_type::realize_matrix_view(R);

        // TODO: is alignment necessary?
        A_VC_STAR.AlignWith(R);

        // Alltoall within process rows
        A_VC_STAR = A;

        // Global size of the result of the Local Gemm that follows
        sketch_of_A_STAR_STAR.Resize(R.Height(),
                                     A_VC_STAR.Width());

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::NORMAL,
                   value_type(1),
                   R.LockedMatrix(),
                   A_VC_STAR.LockedMatrix(),
                   value_type(0),
                   sketch_of_A_STAR_STAR.Matrix());

        // Reduce-scatter within process grid
        sketch_of_A.SumScatterFrom(sketch_of_A_STAR_STAR);

    }


#endif // OPTIMIZED


////////////////////////////////////////////////////////////////////////////////
#ifdef OPTIMIZED // OPTIMIZED

    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {


        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::MR, elem::STAR> R1(grid);
        elem::DistMatrix<value_type>
            A_Left(grid),
            A_Right(grid),
            A0(grid),
            A1(grid),
            A2(grid);
        elem::DistMatrix<value_type, elem::MC, elem::STAR>
            A1_MC_STAR(grid);

        // Zero sketch_of_A
        elem::Zero(sketch_of_A);

        // TODO: are alignments necessary?
        R1.AlignWith(sketch_of_A);
        A1_MC_STAR.AlignWith(sketch_of_A);

        elem::LockedPartitionRight
        ( A,
          A_Left, A_Right, 0 );

        int blocksize = get_blocksize();
        int base = 0;
        while (A_Right.Width() > 0) {

            int b = std::min(A_Right.Width(), blocksize);
            data_type::realize_matrix_view(R1, 0,                   base,
                                               sketch_of_A.Width(), b);

            elem::RepartitionRight
            ( A_Left, /**/      A_Right,
              A0,     /**/ A1,  A2,      b );

            // Allgather within process rows
            A1_MC_STAR = A1;

            // Local Gemm
            base::Gemm(elem::NORMAL,
                       elem::TRANSPOSE,
                       value_type(1),
                       A1_MC_STAR.LockedMatrix(),
                       R1.LockedMatrix(),
                       value_type(1),
                       sketch_of_A.Matrix());

            base = base + b;

            elem::SlidePartitionRight
            ( A_Left,     /**/ A_Right,
              A0,     A1, /**/ A2 );

        }
    }

#else

    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {


        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::MR, elem::STAR> R(grid);
        elem::DistMatrix<value_type, elem::MC, elem::STAR>
            A_MC_STAR(grid);

        // TODO: are alignments necessary?
        R.AlignWith(sketch_of_A);
        A_MC_STAR.AlignWith(sketch_of_A);

        data_type::realize_matrix_view(R);

        // Allgather within process rows
        A_MC_STAR = A;

        // Zero sketch_of_A
        elem::Zero(sketch_of_A);

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::TRANSPOSE,
                   value_type(1),
                   A_MC_STAR.LockedMatrix(),
                   R.LockedMatrix(),
                   value_type(1),
                   sketch_of_A.Matrix());
    }

#endif // OPTIMIZED


////////////////////////////////////////////////////////////////////////////////
#ifdef OPTIMIZED // OPTIMIZED

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

        // Zero sketch_of_A
        elem::Zero(sketch_of_A);

        // TODO: are alignments necessary?
        R1.AlignWith(sketch_of_A);
        A1Trans_MR_STAR.AlignWith(sketch_of_A);

        elem::LockedPartitionDown
        ( A,
          A_Top, A_Bottom, 0 );

        int blocksize = get_blocksize();
        int base = 0;
        while (A_Bottom.Height() > 0) {

            int b = std::min(A_Bottom.Height(), blocksize);
            data_type::realize_matrix_view(R1, 0,                   base,
                                               sketch_of_A.Height(), b);

            elem::RepartitionDown
             ( A_Top,    A0,
               /**/      /**/
                         A1,
               A_Bottom, A2, b );


            // Global size of the target of Allgather that follows
            A1Trans_MR_STAR.Resize(A1.Width(),
                                   A1.Height());

            // Allgather within process columns
            // TODO: Describe cache benefits from transposition:
            //       why not simply use A1[STAR, MR]?
            A1.TransposeColAllGather(A1Trans_MR_STAR);

            // Local Gemm
            base::Gemm(elem::NORMAL,
                       elem::TRANSPOSE,
                       value_type(1),
                       R1.LockedMatrix(),
                       A1Trans_MR_STAR.LockedMatrix(),
                       value_type(1),
                       sketch_of_A.Matrix());

            base = base + b;

            elem::SlidePartitionDown
            ( A_Top,    A0,
                        A1,
              /**/      /**/
              A_Bottom, A2 );
        }
    }

#else

    void outer_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::MC, elem::STAR> R(grid);
        elem::DistMatrix<value_type, elem::MR, elem::STAR>
            ATrans_MR_STAR(grid);

        // TODO: are alignments necessary?
        R.AlignWith(sketch_of_A);
        ATrans_MR_STAR.AlignWith(sketch_of_A);

        data_type::realize_matrix_view(R);

        // Zero sketch_of_A
        elem::Zero(sketch_of_A);

        // Global size of the target of Allgather that follows
        ATrans_MR_STAR.Resize(A.Width(),
                              A.Height());

        // Allgather within process columns
        // TODO: Describe cache benefits from transposition:
        //       why not simply use A1[STAR, MR]?
        A.TransposeColAllGather(ATrans_MR_STAR);

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::TRANSPOSE,
                   value_type(1),
                   R.LockedMatrix(),
                   ATrans_MR_STAR.LockedMatrix(),
                   value_type(1),
                   sketch_of_A.Matrix());
    }

#endif // OPTIMIZED


////////////////////////////////////////////////////////////////////////////////
#ifdef OPTIMIZED // OPTIMIZED

    void matrix_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::MR> R1(grid);
        elem::DistMatrix<value_type>
            sketch_of_A_Left(grid),
            sketch_of_A_Right(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid);
        elem::DistMatrix<value_type, elem::MC, elem::STAR>
            sketch_of_A_temp(grid);

        // TODO: are alignments necessary?
        R1.AlignWith(sketch_of_A);
        sketch_of_A_temp.AlignWith(sketch_of_A);

        elem::PartitionRight
        ( sketch_of_A,
          sketch_of_A_Left, sketch_of_A_Right, 0 );

        int blocksize = get_blocksize();
        int base = 0;
        while (sketch_of_A_Right.Width() > 0) {

            int b = std::min(sketch_of_A_Right.Width(), blocksize);
            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Width());

            elem::RepartitionRight
            ( sketch_of_A_Left, /**/               sketch_of_A_Right,
              sketch_of_A0,     /**/ sketch_of_A1, sketch_of_A2,      b );

            // Global size of the result of the Local Gemm that follows
            sketch_of_A_temp.Resize(sketch_of_A.Height(),
                                    R1.Height());

            // Local Gemm
            base::Gemm(elem::NORMAL,
                       elem::TRANSPOSE,
                       value_type(1),
                       A.LockedMatrix(),
                       R1.LockedMatrix(),
                       sketch_of_A_temp.Matrix());

            // Reduce-scatter within row communicators
            sketch_of_A1.RowSumScatterFrom(sketch_of_A_temp);

            base = base + b;

            elem::SlidePartitionRight
            ( sketch_of_A_Left,               /**/ sketch_of_A_Right,
              sketch_of_A0,     sketch_of_A1, /**/ sketch_of_A2 );

        }
    }

#else

    void matrix_panel_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::MR> R(grid);
        elem::DistMatrix<value_type, elem::MC, elem::STAR>
            sketch_of_A_temp(grid);

        // TODO: are alignments necessary?
        R.AlignWith(A);
        sketch_of_A_temp.AlignWith(A);

        data_type::realize_matrix_view(R);

        // Global size of the result of the Local Gemm that follows
        sketch_of_A_temp.Resize(sketch_of_A.Height(),
                                sketch_of_A.Width());

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::TRANSPOSE,
                   value_type(1),
                   A.LockedMatrix(),
                   R.LockedMatrix(),
                   sketch_of_A_temp.Matrix());

        // Reduce-scatter within row communicators
        sketch_of_A.RowSumScatterFrom(sketch_of_A_temp);

    }


#endif // OPTIMIZED


////////////////////////////////////////////////////////////////////////////////
#ifdef OPTIMIZED // OPTIMIZED

#ifdef EXPERIMENTAL // EXPERIMENTAL

    void panel_matrix_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::MC> R1(grid);
        elem::DistMatrix<value_type>
            sketch_of_A_Top(grid),
            sketch_of_A_Bottom(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid);
        elem::DistMatrix<value_type, elem::MR, elem::STAR>
            sketch_of_A_temp(grid);

        // TODO: are alignments necessary?
        R1.AlignWith(A);
        sketch_of_A_temp.AlignWith(A);

        elem::PartitionDown
        ( sketch_of_A,
          sketch_of_A_Top, sketch_of_A_Bottom, 0 );

        int blocksize = get_blocksize();
        int base = 0;
        while (sketch_of_A_Bottom.Height() > 0) {

            int b = std::min(sketch_of_A_Bottom.Height(), blocksize);

            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Height());

            elem::RepartitionDown
            ( sketch_of_A_Top,     sketch_of_A0,
              /**/                 /**/
                                   sketch_of_A1,
              sketch_of_A_Bottom, sketch_of_A2, b );

            // Global size of the result of the Local Gemm that follows
            sketch_of_A_temp.Resize(sketch_of_A1.Height(),
                                    sketch_of_A1.Width());

            // Local Gemm
            // A.T[MR, MC] * R1.T[MC, STAR] = (A.T * R1.T)[MR, STAR]:
            base::Gemm(elem::TRANSPOSE,
                       elem::TRANSPOSE,
                       value_type(1),
                       A.LockedMatrix(),
                       R1.LockedMatrix(),
                       sketch_of_A_temp.Matrix());

            // TODO: Revisit the transposition logic
            // Reduce-scatter within column communicators
            // TODO: Describe cache benefits from transposition of terms
            //       and implicit transposition of result after the summation

            sketch_of_A1.TransposeColSumScatterUpdate(value_type(1),
                sketch_of_A_temp);

            base = base + b;

            elem::SlidePartitionDown
            ( sketch_of_A_Top,    sketch_of_A0,
                                  sketch_of_A1,
              /**/                /**/
              sketch_of_A_Bottom, sketch_of_A2 );
        }
    }

#else

    void panel_matrix_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::MC> R1(grid);
        elem::DistMatrix<value_type>
            sketch_of_A_Top(grid),
            sketch_of_A_Bottom(grid),
            sketch_of_A0(grid),
            sketch_of_A1(grid),
            sketch_of_A2(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::MR>
            sketch_of_A_temp(grid);

        // TODO: are alignments necessary?
        R1.AlignWith(A);
        sketch_of_A_temp.AlignWith(A);

        elem::PartitionDown
        ( sketch_of_A,
          sketch_of_A_Top, sketch_of_A_Bottom, 0 );

        int blocksize = get_blocksize();
        int base = 0;
        while (sketch_of_A_Bottom.Height() > 0) {

            int b = std::min(sketch_of_A_Bottom.Height(), blocksize);

            data_type::realize_matrix_view(R1, base, 0,
                                               b,    A.Height());

            elem::RepartitionDown
            ( sketch_of_A_Top,     sketch_of_A0,
              /**/                 /**/
                                   sketch_of_A1,
              sketch_of_A_Bottom, sketch_of_A2, b );

            // Global size of the result of the Local Gemm that follows
            sketch_of_A_temp.Resize(R1.Height(),
                                    A.Width());

            // Local Gemm
            base::Gemm(elem::NORMAL,
                       elem::NORMAL,
                       value_type(1),
                       R1.LockedMatrix(),
                       A.LockedMatrix(),
                       sketch_of_A_temp.Matrix());

            // Reduce-scatter within column communicators
            sketch_of_A1.ColSumScatterFrom(sketch_of_A_temp);

            // Reduce-scatter within column communicators
            // sketch_of_A1.ColSumScatterUpdate(value_type(1),
            //    sketch_of_A_temp);

            base = base + b;

            elem::SlidePartitionDown
            ( sketch_of_A_Top,    sketch_of_A0,
                                  sketch_of_A1,
              /**/                /**/
              sketch_of_A_Bottom, sketch_of_A2 );
        }
    }

#endif // EXPERIMENTAL

#else

    void panel_matrix_gemm(const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag) const {

        const elem::Grid& grid = A.Grid();

        elem::DistMatrix<value_type, elem::STAR, elem::MC> R(grid);
        elem::DistMatrix<value_type, elem::STAR, elem::MR>
            sketch_of_A_temp(grid);

        // TODO: is alignment necessary?
        sketch_of_A_temp.AlignWith(A);

        data_type::realize_matrix_view(R);

        // Global size of the result of the Local Gemm that follows
        sketch_of_A_temp.Resize(R.Height(),
                                A.Width());

        // Local Gemm
        base::Gemm(elem::NORMAL,
                   elem::NORMAL,
                   value_type(1),
                   R.LockedMatrix(),
                   A.LockedMatrix(),
                   sketch_of_A_temp.Matrix());

        // Reduce-scatter within column communicators
        sketch_of_A.ColSumScatterFrom(sketch_of_A_temp);

    }

#endif // OPTIMIZED

    void sketch_gemm(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::rowwise_tag tag) const {

        const int sketch_height = sketch_of_A.Height();
        const int sketch_width  = sketch_of_A.Width();
        const int width         = A.Width();

        const double factor = get_factor();

#ifdef TESTED_ROWWISE
        TESTED_ROWWISE
#else
        if((sketch_height * factor <= width) &&
            (sketch_width * factor <= width))
            inner_panel_gemm(A, sketch_of_A, tag);
        else if((sketch_height >= width * factor) &&
            (sketch_width >= width * factor))
            outer_panel_gemm(A, sketch_of_A, tag);
        else
            matrix_panel_gemm(A, sketch_of_A, tag);
#endif
    }


    void sketch_gemm(const matrix_type& A,
        output_matrix_type& sketch_of_A,
        skylark::sketch::columnwise_tag tag) const {

        const int sketch_height = sketch_of_A.Height();
        const int sketch_width  = sketch_of_A.Width();
        const int height         = A.Height();

        const double factor = get_factor();

#ifdef TESTED_COLUMNWISE
        TESTED_COLUMNWISE
#else
        if((sketch_height * factor <= height) &&
            (sketch_width * factor <= height))
            inner_panel_gemm(A, sketch_of_A, tag);
        else if((sketch_height >= height * factor) &&
            (sketch_width >= height * factor))
            outer_panel_gemm(A, sketch_of_A, tag);
        else
            panel_matrix_gemm(A, sketch_of_A, tag);
#endif
    }

    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::rowwise_tag tag) const {

        sketch_gemm(A, sketch_of_A, tag);
    }


    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          skylark::sketch::columnwise_tag tag) const {

        sketch_gemm(A, sketch_of_A, tag);
    }


////////////////////////////////////////////////////////////////////////////////

#else // HP_DENSE_TRANSFORM_ELEMENTAL_MC_MR

////////////////////////////////////////////////////////////////////////////////

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for distributed input/output [MC, MR] and columnwise.
     */
    void apply_impl_dist (const matrix_type& A,
                          output_matrix_type& sketch_of_A,
                          columnwise_tag) const {
        // std::cout << "Reference Impl" << std::endl;
        elem::DistMatrix<value_type> S(data_type::_S, data_type::_N);

        for(int j_loc = 0; j_loc < S.LocalWidth(); ++j_loc) {
            int j = S.RowShift() + S.RowStride() * j_loc;
            for (int i_loc = 0; i_loc < S.LocalHeight(); ++i_loc) {
                int i = S.ColShift() + S.ColStride() * i_loc;
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.SetLocal(i_loc, j_loc, data_type::scale * sample);
            }
        }
        base::Gemm (elem::NORMAL,
                    elem::NORMAL,
                    1.0,
                    S,
                    A,
                    0.0,
                    sketch_of_A);

    }

    /**
     * Apply the sketching transform that is described in by the sketch_of_A.
     * Implementation for distributed input/output [MC, MR] and rowwise.
     */
    void apply_impl_dist(const matrix_type& A,
                         output_matrix_type& sketch_of_A,
                         rowwise_tag) const {
        // std::cout << "Reference Impl" << std::endl;
        elem::DistMatrix<value_type> S(data_type::_S, data_type::_N);

        for(int j_loc = 0; j_loc < S.LocalWidth(); ++j_loc) {
            int j = S.RowShift() + S.RowStride() * j_loc;
            for (int i_loc = 0; i_loc < S.LocalHeight(); ++i_loc) {
                int i = S.ColShift() + S.ColStride() * i_loc;
                value_type sample =
                    data_type::random_samples[j * data_type::_S + i];
                S.SetLocal(i_loc, j_loc, data_type::scale * sample);
            }
        }
        base::Gemm (elem::NORMAL,
                    elem::TRANSPOSE,
                    1.0,
                    A,
                    S,
                    0.0,
                    sketch_of_A);

    }


#endif

};

} } /** namespace skylark::sketch */

#endif // SKYLARK_DENSE_TRANSFORM_ELEMENTAL_MC_MR_HPP
