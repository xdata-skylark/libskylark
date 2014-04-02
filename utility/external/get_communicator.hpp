#ifndef SKYLARK_GET_COMMUNICATOR_HPP
#define SKYLARK_GET_COMMUNICATOR_HPP

// TODO: Replace with Skylark specific exceptions.
#include <exception>

#include "../../config.h"

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif


#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif


#if SKYLARK_HAVE_BOOST
#include <boost/mpi.hpp>
#endif


namespace skylark { namespace utility {

// namespace alias
namespace mpi = boost::mpi;

/** Define a base type to describe a matrix */
struct matrix_type_tag {};

/** Convenient macro for generating type mapping template specializations */
#define TYPE_MAPPER(FROM_TYPE, TO_TYPE) \
template<> \
struct type_mapper_t<FROM_TYPE> { \
    typedef TO_TYPE type; \
};

/** Empty type mapping */
template<typename T>
struct type_mapper_t {};



#if SKYLARK_HAVE_ELEMENTAL

/** Denote a local dense matrix type as defined in Elemental */
struct elemental_matrix_type_tag : matrix_type_tag {};
/** Denote a distributed dense matrix type as defined in Elemental */
struct elemental_distributed_matrix_type_tag : matrix_type_tag {};

/** Convenient type aliases for most common matrix objects in Elemental */
typedef elem::Matrix<double>            dense_matrix_t;
typedef elem::DistMatrix<double, elem::MC,   elem::MR>
dist_dense_matrix_t;
typedef elem::DistMatrix<double, elem::VC,   elem::STAR>
dist_VC_STAR_dense_matrix_t;
typedef elem::DistMatrix<double, elem::VR,   elem::STAR>
dist_VR_STAR_dense_matrix_t;
typedef elem::DistMatrix<double, elem::STAR, elem::VC>
dist_STAR_VC_dense_matrix_t;
typedef elem::DistMatrix<double, elem::STAR, elem::VR>
dist_STAR_VR_dense_matrix_t;

/** Type mapping template specializations */
TYPE_MAPPER(dense_matrix_t,              elemental_matrix_type_tag)
TYPE_MAPPER(dist_dense_matrix_t,         elemental_distributed_matrix_type_tag)
TYPE_MAPPER(dist_VC_STAR_dense_matrix_t, elemental_distributed_matrix_type_tag)
TYPE_MAPPER(dist_VR_STAR_dense_matrix_t, elemental_distributed_matrix_type_tag)
TYPE_MAPPER(dist_STAR_VC_dense_matrix_t, elemental_distributed_matrix_type_tag)
TYPE_MAPPER(dist_STAR_VR_dense_matrix_t, elemental_distributed_matrix_type_tag)

/** Type mapping template specializations */
TYPE_MAPPER(const dense_matrix_t,              elemental_matrix_type_tag)
TYPE_MAPPER(const dist_dense_matrix_t,         elemental_distributed_matrix_type_tag)
TYPE_MAPPER(const dist_VC_STAR_dense_matrix_t, elemental_distributed_matrix_type_tag)
TYPE_MAPPER(const dist_VR_STAR_dense_matrix_t, elemental_distributed_matrix_type_tag)
TYPE_MAPPER(const dist_STAR_VC_dense_matrix_t, elemental_distributed_matrix_type_tag)
TYPE_MAPPER(const dist_STAR_VR_dense_matrix_t, elemental_distributed_matrix_type_tag)


#endif // SKYLARK_HAVE_ELEMENTAL


#if SKYLARK_HAVE_COMBBLAS

/** Denote a distributed sparse matrix type as defined in CombBLAS */
struct combblas_distributed_matrix_type_tag : matrix_type_tag {};

/** Convenient type aliases for most common matrix objects in CombBLAS */
typedef FullyDistVec<size_t, double>    dist_vector_t;
typedef SpDCCols<size_t, double>        col_t;
typedef SpParMat<size_t, double, col_t> dist_sparse_matrix_t;

/** Type mapping template specializations */
TYPE_MAPPER(dist_sparse_matrix_t,        combblas_distributed_matrix_type_tag)
TYPE_MAPPER(const dist_sparse_matrix_t,        combblas_distributed_matrix_type_tag)


#endif // SKYLARK_HAVE_COMBBLAS


#if SKYLARK_HAVE_ELEMENTAL && SKYLARK_HAVE_BOOST

template <typename T>
mpi::communicator get_communicator(T& A,
    elemental_matrix_type_tag) {
    return mpi::communicator(MPI_COMM_SELF, mpi::comm_duplicate);
}


template <typename T>
mpi::communicator get_communicator(T& A,
    elemental_distributed_matrix_type_tag) {
    return mpi::communicator(A.DistComm(), mpi::comm_duplicate);
}

#endif // SKYLARK_HAVE_ELEMENTAL and SKYLARK_HAVE_BOOST


#if SKYLARK_HAVE_COMBBLAS && SKYLARK_HAVE_BOOST

template <typename T>
mpi::communicator get_communicator(T& A,
    combblas_distributed_matrix_type_tag) {
    return mpi::communicator(A.getcommgrid()->GetWorld(), mpi::comm_duplicate);
}

#endif // SKYLARK_HAVE_COMBBLAS AND SKYLARK_HAVE_BOOST


#if SKYLARK_HAVE_BOOST

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


/**
 * Returns the boost MPI communicator of the supplied matrix A.
 *
 * First maps the type of A to one of the larger, library-dependent classes of
 * matrix types as previosusly identified and then forwards to the respective,
 * auxiliary 2-argument routine.
 */
template <typename T>
mpi::communicator get_communicator(T& A) {
    typename type_mapper_t<T>::type type;
    return get_communicator(A, type);
}

#endif // SKYLARK_HAVE_BOOST

} } /** namespace skylark::utility */

#endif // SKYLARK_GET_COMMUNICATOR_HPP
