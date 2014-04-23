#ifndef SKYLARK_GET_COMMUNICATOR_HPP
#define SKYLARK_GET_COMMUNICATOR_HPP

// TODO: Replace with Skylark specific exceptions.
#include <exception>

#include "../config.h"
#include "../base/base.hpp"

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif


#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif

#include <boost/mpi.hpp>


namespace skylark { namespace utility {

// namespace alias
namespace mpi = boost::mpi;

template<typename T>
mpi::communicator get_communicator(const base::sparse_matrix_t<T>& A,
    mpi::comm_create_kind kind = mpi::comm_attach) {
    return mpi::communicator(MPI_COMM_SELF, kind);
}

#if SKYLARK_HAVE_ELEMENTAL

template<typename T>
mpi::communicator get_communicator(const elem::Matrix<T>& A,
    mpi::comm_create_kind kind = mpi::comm_attach) {
    return mpi::communicator(MPI_COMM_SELF, kind);
}

template<typename T, elem::Distribution U, elem::Distribution V>
mpi::communicator get_communicator(const elem::DistMatrix<T, U, V>& A,
    mpi::comm_create_kind kind = mpi::comm_attach) {
    return mpi::communicator(A.DistComm(), kind);
}

#endif // SKYLARK_HAVE_ELEMENTAL


#if SKYLARK_HAVE_COMBBLAS

template<typename IT, typename T, typename S>
mpi::communicator get_communicator(const SpParMat<IT, T, S>& A,
    mpi::comm_create_kind kind = mpi::comm_attach) {
    return mpi::communicator(A.getcommgrid()->GetWorld(), kind);
}

#endif // SKYLARK_HAVE_COMBBLAS


/**
 * Utility routine checking if the argument boost MPI communicators are either
 * identical or congruent and returns true in this case;
 * returns false otherwise.
 */
bool compatible(const mpi::communicator& comm_A,
    const mpi::communicator& comm_B) {
    int error_code, result;
    error_code  = MPI_Comm_compare((MPI_Comm)comm_A, (MPI_Comm)comm_B, &result);
    if(error_code != MPI_SUCCESS) {
        throw std::runtime_error("MPI failure during call");
    }

    if (result == MPI_IDENT || result == MPI_CONGRUENT) {
        return true;
    }
    return false;
}

} } /** namespace skylark::utility */

#endif // SKYLARK_GET_COMMUNICATOR_HPP
