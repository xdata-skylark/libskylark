/*
 * replicate_matrix.hpp
 *
 *  Created on: Aug 14, 2015
 *      Author: chander
 */

#ifndef REPLICATE_MATRIX_HPP_
#define REPLICATE_MATRIX_HPP_

namespace skylark {
namespace utility {

#include <El.hpp>


template<typename T>
inline void ReplicateDistMatrix(El::DistMatrix<T>& A, int ntimes) {

    El::Int commRank = El::mpi::Rank( El::mpi::COMM_WORLD );

    std::vector<El::DistMatrix<T> > tempMergeA( ntimes*2, El::DistMatrix<T>(A.Grid()) );
    El::Zeros(tempMergeA[0], ntimes*A.Height(), A.Width());

    for(int i = 0 ; i < ( 2*(ntimes-1) ) ; i++)
    {
        if(i%2)
                El::Copy(A, tempMergeA[i]) ;
        else
                El::PartitionDown(tempMergeA[i], tempMergeA[i+1], tempMergeA[i+2], A.Height()) ;
    }

    El::Copy(A, tempMergeA[2*(ntimes-1)]);
    for(int i = 0 ; i < ( 2*(ntimes-1) ) ; i++)
        if(!(i%2))
                tempMergeA[i] = El::Merge2x1(tempMergeA[i+1], tempMergeA[i+2]) ;

  A = tempMergeA[0] ;
}

}

}




#endif /* REPLICATE_MATRIX_HPP_ */
