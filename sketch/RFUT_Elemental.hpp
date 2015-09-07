#ifndef SKYLARK_RFUT_ELEMENTAL_HPP
#define SKYLARK_RFUT_ELEMENTAL_HPP

#include <boost/mpi.hpp>
#include "spi/include/kernel/memory.h"

namespace skylark { namespace sketch {

/**
 * Specialization for [*, SOMETHING]
 */
template < typename ValueType,
           typename FUT,
           El::Distribution RowDist,
           typename ValueDistributionType>
struct RFUT_t<
    El::DistMatrix<ValueType, El::STAR, RowDist>,
    FUT,
    ValueDistributionType> :
        public RFUT_data_t<ValueDistributionType> {
    // Typedef value, matrix, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::Matrix<ValueType> local_type;
    typedef El::DistMatrix<ValueType, El::STAR, RowDist> matrix_type;
    typedef El::DistMatrix<ValueType,
                             El::STAR, RowDist> output_matrix_type;
    typedef ValueDistributionType value_distribution_type;
    typedef RFUT_data_t<ValueDistributionType> data_type;

    /**
     * Regular constructor
     */
    RFUT_t(int N, base::context_t& context)
        : data_type (N, context) {

    }

    /**
     * Copy constructor
     */
    RFUT_t (RFUT_t<matrix_type,
                   FUT,
                   value_distribution_type>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    RFUT_t(const data_type& other_data) :
        data_type(other_data) {}

    /**
     * Apply the transform that is described in by the mixed_A.
     * mixed_A can be the same as A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A,
        output_matrix_type& mixed_A,
        Dimension dimension) const {
        switch (RowDist) {
        case El::VC:
        case El::VR:
            try {
                apply_impl_vdist(A, mixed_A, dimension);
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
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the columns.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& mixed_A,
                           skylark::sketch::columnwise_tag) const {
        // TODO verify that A has the correct size

        // TODO no need to create FUT everytime...
        FUT T(data_type::_N);

        // Scale
        const local_type& local_A = A.LockedMatrix();
        local_type& local_TA = mixed_A.Matrix();
        value_type scale = T.scale();
        for (int j = 0; j < local_A.Width(); j++)
            for (int i = 0; i < data_type::_N; i++)
                local_TA.Set(i, j,
                    scale * data_type::D[i] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_TA, skylark::sketch::columnwise_tag());
    }


};

/**
 * Specialization for [SOMETHING, *]
 */
template < typename ValueType,
           typename FUT,
           El::Distribution RowDist,
           typename ValueDistributionType>
struct RFUT_t<
    El::DistMatrix<ValueType, RowDist, El::STAR>,
    FUT,
    ValueDistributionType> :
        public RFUT_data_t<ValueDistributionType> {
    // Typedef value, matrix, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::Matrix<ValueType> local_type;
    typedef El::DistMatrix<ValueType, RowDist, El::STAR> matrix_type;
    typedef El::DistMatrix<ValueType, RowDist, El::STAR> output_matrix_type;
    typedef El::DistMatrix<ValueType, El::STAR, RowDist> intermediate_type;
    /**< Intermediate type for columnwise applications */
    typedef ValueDistributionType value_distribution_type;
    typedef RFUT_data_t<ValueDistributionType> data_type;

    /**
     * Regular constructor
     */
    RFUT_t(int N, base::context_t& context)
        : data_type (N, context) {

    }

    /**
     * Copy constructor
     */
    RFUT_t (RFUT_t<matrix_type,
                   FUT,
                   value_distribution_type>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    RFUT_t(const data_type& other_data) :
           data_type(other_data) {}

    /**
     * Apply the transform that is described in by the mixed_A.
     */
    template <typename Dimension>
    void apply(const matrix_type& A,
               output_matrix_type& mixed_A,
               Dimension dimension) const {

        switch (RowDist) {
            case El::VC:
            case El::VR:
                try {
                    apply_impl_vdist(A, mixed_A, dimension);
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

    template <typename Dimension>
    void apply_inverse(const matrix_type& A,
                       output_matrix_type& mixed_A,
                       Dimension dimension) const {

        switch (RowDist) {
            case El::VC:
            case El::VR:
                try {
                    apply_inverse_impl_vdist(A, mixed_A, dimension);
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
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the rows.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& mixed_A,
                           skylark::sketch::rowwise_tag) const {
        // TODO verify that A has the correct size

        FUT T(data_type::_N);

        // Scale
        const local_type& local_A = A.LockedMatrix();
        local_type& local_TA = mixed_A.Matrix();
        value_type scale = T.scale(local_A);
        for (int j = 0; j < data_type::_N; j++)
            for (int i = 0; i < local_A.Height(); i++)
                local_TA.Set(i, j,
                    scale * data_type::D[j] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_TA, skylark::sketch::rowwise_tag());
    }

    /**
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the columns.
     */
    void apply_impl_vdist (const matrix_type& A,
                           output_matrix_type& mixed_A,
                           skylark::sketch::columnwise_tag) const {
        // TODO verify that A has the correct size
        // TODO A and mixed_A have to match

        FUT T(data_type::_N);

        // Rearrange matrix
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Scale
        local_type& local_A = inter_A.Matrix();
        value_type scale = T.scale(local_A);
        for (int j = 0; j < local_A.Width(); j++)
            for (int i = 0; i < data_type::_N; i++)
                local_A.Set(i, j,
                    scale * data_type::D[i] * local_A.Get(i, j));

        // Apply underlying transform
        T.apply(local_A, skylark::sketch::columnwise_tag());

        // Rearrange back
        mixed_A = inter_A;
    }

    /**
     * Apply the transform to compute mixed_A.
     * Implementation for the application on the columns.
     */
    void apply_inverse_impl_vdist  (const matrix_type& A,
                                    output_matrix_type& mixed_A,
                                    skylark::sketch::columnwise_tag) const {

        FUT T(data_type::_N);

        // TODO verify that A has the correct size
        // TODO A and mixed_A have to match

        // Rearrange matrix
        intermediate_type inter_A(A.Grid());
        inter_A = A;

        // Apply underlying transform
        local_type& local_A = inter_A.Matrix();
        T.apply_inverse(local_A, skylark::sketch::columnwise_tag());

        // Scale
        value_type scale = T.scale(local_A);
        for (int j = 0; j < local_A.Width(); j++)
            for (int i = 0; i < data_type::_N; i++)
                local_A.Set(i, j,
                    scale * data_type::D[i] * local_A.Get(i, j));

        // Rearrange back
        mixed_A = inter_A;
    }

};

/**
 * Specialization for [MC, MR]
 */
template < typename ValueType,
         typename FUT,
         typename ValueDistributionType>
struct RFUT_t<
         El::DistMatrix<ValueType>,
         FUT,
         ValueDistributionType> :
         public RFUT_data_t<ValueDistributionType> {
    // Typedef value, matrix, distribution and transform data types
    // so that we can use them regularly and consistently.
    typedef ValueType value_type;
    typedef El::Matrix<ValueType> local_type;
    typedef El::DistMatrix<ValueType> matrix_type;
    typedef El::DistMatrix<ValueType> output_matrix_type;
    typedef ValueDistributionType value_distribution_type;
    typedef RFUT_data_t<ValueDistributionType> data_type;

    /**
     * Regular constructor
     */
    RFUT_t(int N, base::context_t& context)
        : data_type (N, context) {
    }

    /**
     * Copy constructor
     */
    RFUT_t (RFUT_t<matrix_type,
                   FUT,
                   value_distribution_type>& other) :
        data_type(other) {}

    /**
     * Constructor from data
     */
    RFUT_t(const data_type& other_data) :
        data_type(other_data) {}

    /**
     * Apply the transform that is described in by the mixed_A.
     * mixed_A can be the same as A.
     */
    template <typename Dimension>
    void apply (const matrix_type& A, output_matrix_type& mixed_A,
    		base::context_t& context, Dimension dimension) const
    {
        try
        {
            apply_impl_vdist(A, mixed_A, context, dimension);
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
   /**
    * Apply the transform to compute mixed_A.
    * Implementation for the application on the columns.
    */
    void apply_impl_vdist (const matrix_type& A,
              output_matrix_type& mixed_A,
              base::context_t& context, 
              skylark::sketch::columnwise_tag) const
    {
        FUT T( data_type::_N );
        El::Int commSize = El::mpi::Size( El::mpi::COMM_WORLD );
        El::Int commRank = El::mpi::Rank( El::mpi::COMM_WORLD );

        //Scale the Matrix with the Radhemacher diagonal matrix
        //and the DCT-II scaling factor

        El::DistMatrix<ValueType, El::VR, El::STAR> randDiagMatrix(data_type::_N, 1, A.Grid()) ;
        El::ThreeValued(randDiagMatrix, data_type::_N, 1, value_type(1.0) ) ;


        boost::random::uniform_int_distribution<int> distribution(0, data_type::_N - 1);
        std::vector<int> samples =
            context.generate_random_samples_array(mixed_A.Height(), distribution);

        double scale = std::sqrt((double)data_type::_N / (double)mixed_A.Height());

       /*
        * Here we do the Batchwise split and transformation as follows ::
        * 1. We take batches of column sizes up to number of MPI processes
        *    since those number of columns should fit in memory at the
        *    minimum for translation.
        * 2. Each [MC, MR] batch is translated to [STAR,VR].
        * 3. This [STAR, VR] is transformed by the FUT instance.
        * 4. The  [STAR, VR] batch is then translated back into the original matrix.
        */
        El::DistMatrix<ValueType, El::STAR, El::VR> ABatch (A.Grid());
        El::DistMatrix<ValueType, El::STAR, El::VR> SABatch (A.Grid());
       /*
        * The idea of assigning a batch size is as follows:
        * 1. You divide the heap memory available per node in bytes by the
        *    row height.
        * 2. The minimum batchsize is obtained and broadcasted to all nodes.
        * 3. If the batchsize is smaller than the number of processes in
        *    in the system, set the batchsize to the number of processes.
        * 4. If the batchsize is greater than the Matrix width, then set the
        *    batchsize to the matrix width.
        */

        uint64_t heapavail ;
        Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
        El::Int batchSize = heapavail / (A.Height() + mixed_A.Height()) ;
        const El::Int minBatchSize = El::mpi::AllReduce(batchSize, El::mpi::MIN,
        El::mpi::COMM_WORLD) ;
        batchSize = minBatchSize ;

        batchSize = ((batchSize * commSize) / 32 )  ;
        if(batchSize <= commSize)
            batchSize = commSize ;
        else if(batchSize >= A.Width())
            batchSize = A.Width() ;
        else
        {
           if( ((A.Height() + mixed_A.Height()) / batchSize) > A.Width())
           {
             int scaleFactor = ( (A.Height() + mixed_A.Height()) / (batchSize * A.Width()) ) + 1 ;
             batchSize -= 0.02 * scaleFactor *A.Width() ;
           }
        }

        El::Int numBatches = A.Width() / batchSize ;
        if(A.Width() % batchSize)
            numBatches+= 1 ;

        if (commRank == 0)
            std::cout << "Batchsize is :: " << batchSize << "\t and numBatches is :: " << numBatches << std::endl ;

        El::Range<El::Int> rowRange(0,A.Height()) ;
        El::Range<El::Int> sarowRange(0,mixed_A.Height()) ;
        
        //Outer loop for batchwise conversion
        for (El::Int cnt = 0; cnt < numBatches ; cnt++)
        {
            El::Int colStart = cnt*batchSize ;
            El::Int colEnd   = (cnt+1)*batchSize ;

            ABatch.Resize( A.Height(), batchSize );
            SABatch.Resize(mixed_A.Height(), batchSize ) ;

            if(cnt == (numBatches-1) )
            {
                colEnd = A.Width() ;
                ABatch.Resize(A.Height(), colEnd-colStart) ;
                SABatch.Resize(mixed_A.Height(), colEnd-colStart ) ;
            }

            El::Range<El::Int> colRange(colStart, colEnd) ;

            //Create the [MC, MR] submatrix
            auto ASub  = A(rowRange, colRange) ;
            auto SASub = mixed_A(sarowRange, colRange) ;

            El::DiagonalScale(El::LEFT, El::NORMAL, randDiagMatrix, ASub) ;
            El::Scale(T.scale(), ASub) ;

            //Translate the [MC, MR] to a [STAR, VR] submatrix
            ABatch = ASub ;
            SABatch = SASub ;

            //Apply the transformation
            T.apply(ABatch.Matrix(), skylark::sketch::columnwise_tag());

            for (int j = 0; j < SABatch.LocalWidth() ; j++)
                for (int i = 0; i < SABatch.Height() ; i++) 
                {
                    int row = samples[i];
                    SABatch.Matrix().Set(i, j, ABatch.Matrix().Get(row, j));
                }

            El::Scale(scale, SABatch) ;
            
            //Translate back the [STAR, VR] submatrix into the [MC, MR] matrix
            SASub = SABatch;        
        }
    }


};


} } /** namespace skylark::sketch */

#endif // SKYLARK_RFUT_HPP
