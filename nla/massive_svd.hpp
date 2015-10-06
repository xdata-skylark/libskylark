/*
 * massive_svd.hpp
 *
 *  Created on: Aug 3, 2015
 *      Author: chander
 */

#ifndef MASSIVE_SVD_HPP_
#define MASSIVE_SVD_HPP_

#include <El.hpp>
#include "../sketch/sketch.hpp"

namespace skylark { namespace nla {



/**
 * Parameter structure for approximate SVD
 *
 * oversampling_ratio, oversampling_additive:
 *   given a rank r, the number of columns k in the iterates is
 *   k = oversampling_ratio * r + oversampling_additive
 */
struct massive_svd_params_t : public base::params_t {

    int oversampling_ratio, oversampling_additive;

    massive_svd_params_t(int oversampling_ratio = 10,
        int oversampling_additive = 0,
        int num_iterations = 0,
        bool skip_qr = false,
        bool am_i_printing = false,
        int log_level = 0,
        std::ostream &log_stream = std::cout,
        int debug_level = 0) :
        base::params_t(am_i_printing, log_level, log_stream, debug_level),
        oversampling_ratio(oversampling_ratio),
        oversampling_additive(oversampling_additive) {};
};

/**
 * Massive SVD computation.
 *
 *
 * \param A input matrix
 * \param Uak output: approximate top-k left-singular vectors
 * \param Sak output: approximate top-k singular values
 * \param Vak output: approximate top-k right-singular vectors
 * \param topk target dimension for which top-k singular vectors are extracted.
 * \param params parameter strcture
 */
template <typename InputType, typename UType, typename SType, typename VType>
void MassiveSVD(InputType &A, UType &Uak, SType &Sak, VType &Vak, int topk,
    base::context_t& context,
    massive_svd_params_t params = massive_svd_params_t()) {

    El::mpi::Comm comm = El::mpi::COMM_WORLD;
    const El::Int commRank = El::mpi::Rank( comm );
    const El::Int commSize = El::mpi::Size( comm );

    typedef typename skylark::utility::typer_t<InputType>::value_type
        value_type;

    bool log_lev1 = params.am_i_printing && params.log_level >= 1;
    bool log_lev2 = params.am_i_printing && params.log_level >= 2;

    int m = base::Height(A);
    int n = base::Width(A);

    El::SVDCtrl<value_type> svdparams ;
    svdparams.seqQR         = false ;
    svdparams.valChanRatio  = 1.0 ;
    svdparams.fullChanRatio = 1.0 ;
    svdparams.thresholded   = true ;
    svdparams.relative      = false ;
    svdparams.tol           = 0 ;

    /**
     * Check if sizes match.
     */
    if (topk > std::min(m, n)) 
    {
        std::string msg = "Incompatible matrix dimensions and target rank";
        if (log_lev1)
            params.log_stream << msg << std::endl;
        SKYLARK_THROW_EXCEPTION(base::skylark_exception()
            << base::error_msg(msg));
    }

    /**
     * Code for m >= n ; As of now we only consider approximately square or
     * slightly over-determined systems.
     */
    if (m >= n) {
        int r = std::max(topk,
        		std::min(n,
        		params.oversampling_ratio * topk + params.oversampling_additive));

        /** Apply sketch transformation on the input matrix */
        UType Q(m, r, A.Grid());
        sketch::JLT_t<InputType, UType> Omega(n, r, context);
        Omega.apply(A, Q, sketch::rowwise_tag());

        /**
         * We perform the following steps on the matrix sketch.
         * 1. Compute the product of QtQ = Q^T * Q.
         * 2. Compute the SVD of QtQ = Uq * Sq * Vq^T.
         * 3. Let Vak be the top-k right singular vectors and Sqk be the top-k
         *    singular values obtained from the approximate SVD.
         * 4. Suk = Sqk^(-1/2).
         * 5. Uak  = Q * Vak * Suk.
         * 6. Sak  = Sqk^(1/2).
         */

        /* Step 1 */
        InputType QtQ(r, r, A.Grid()) ;
//        El::Zeros(QtQ, r, r) ;
        base::Gemm(El::TRANSPOSE, El::NORMAL, value_type(1.0), Q, Q, QtQ);

        /* Step 2 */
        SType S(A.Grid()) ;
        VType V(A.Grid()) ;
        El::SVD(QtQ, S, V, svdparams) ;

        /* Step 3 */
        VType Vak(A.Grid()), Vt(A.Grid()) ;
        El::Transpose(V, Vt) ;
        Vak = El::View(Vt, 0, 0, Vt.Height(), topk)  ;

        /* Step 4 */
        InputType Suk(A.Grid()) ;
        std::vector<value_type> sInvVec(topk) ;
        for(int i = 0 ; i < topk ; i++)
            sInvVec.push_back(sqrt(1/S.Get(i,0))) ;
        El::Zeros(Suk, topk, topk) ;
        El::Diagonal(Suk, sInvVec) ;

        /* Step 5 */
        UType Uintk(A.Grid()) ;
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), Q, Vak, Uintk) ;
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), Uintk, Suk, Uak) ;

        /* Step 6 */
        El::Zeros(Sak, topk, topk) ;
        for(int i = 0 ; i < topk ; i++)
            Sak.Set(i,0,sqrt(S.Get(i,0))) ;

        if( commRank == 0)
          std::cout << Sak.Get(0,0) << std::endl ;


        /* Metric 2 */
        VType Vdiff(A.Grid());
        El::Identity(Vdiff, topk, topk) ;
        base::Gemm(El::TRANSPOSE, El::NORMAL, value_type(1.0), Vak, Vak, value_type(-1.0), Vdiff) ;

        /* Metric 3 */

        SType Sallk(A.Grid()), Sall(A.Grid()) ;
        El::SVD(A, Sall, svdparams) ;
        El::Zeros(Sallk, topk, topk) ;
        for(int i = 0 ; i < topk ; i++)
            Sallk.Set(i, 0, Sall.Get(i,0)) ;
        if( commRank == 0)
          std::cout << Sall.Get(0,0) << std::endl ;

        El::Axpy( -1.0, Sak, Sallk );

        const value_type frobNormOfVdiff = El::FrobeniusNorm( Vdiff );
        const value_type singValDiff = El::FrobeniusNorm( Sallk );

        if( commRank == 0)
            std::cout << "||Vk^TVk - I||_F = \t" << frobNormOfVdiff << "\n"
            << "|| sError ||_2 = \t " << singValDiff << std::endl ;

    }

}


} } /** namespace skylark::nla */


#endif /* MASSIVE_SVD_HPP_ */
