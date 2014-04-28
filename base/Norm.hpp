#ifndef SKYLARK_NORM_HPP
#define SKYLARK_NORM_HPP

#include <boost/mpi.hpp>

// Defines a generic Gemm function that recieves a wider set of matrices

#if SKYLARK_HAVE_ELEMENTAL

namespace skylark { namespace base {

template<typename T>
inline elem::Base<T> Nrm2(const elem::Matrix<T>& x) {
    return elem::Nrm2(x);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T>& x) {
    return elem::Nrm2(x);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T, elem::VC, elem::STAR>& x) {
    boost::mpi::communicator comm(x.DistComm(), boost::mpi::comm_attach);
    T local = elem::Nrm2(x.LockedMatrix());
    T snrm = boost::mpi::all_reduce(comm, local * local, std::plus<T>());
    return sqrt(snrm);
}

template<typename T>
inline elem::Base<T> Nrm2(const elem::DistMatrix<T, elem::STAR, elem::STAR>& x) {
    return elem::Nrm2(x.LockedMatrix());
}

} } // namespace skylark::base

#endif // SKYLARK_HAVE_ELEMENTAL

#endif // SKYLARK_NORM_HPP
