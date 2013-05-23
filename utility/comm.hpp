#ifndef COMM_HPP
#define COMM_HPP

#include <elemental.hpp>

#include "config.h"

namespace skylark {
namespace utility {

template<typename T>
void collect_dist_matrix(boost::mpi::communicator& comm, bool here,
    const elem::DistMatrix<T> &DA, elem::Matrix<T> &A) {

    // Technically the following should work for any row and col distributions.
    // But it seems to not work well for VR/VC type distributions.
    // And it is probably not the best way to do it for these distributions
    // anyway.

    elem::AxpyInterface<T> interface;
    interface.Attach(elem::GLOBAL_TO_LOCAL, DA);
    if (here) {
        elem::Zero(A);
        interface.Axpy(1.0, A, 0, 0);
    }
    interface.Detach();
}

template<typename T, elem::Distribution RowDist>
void collect_dist_matrix(boost::mpi::communicator& comm, bool here,
    const elem::DistMatrix<T, RowDist, elem::STAR> &DA,
    elem::Matrix<T> &A) {

    if (RowDist == elem::VR || RowDist == elem::VC) {
        // TODO this is probably the most laziest way to do it.
        //      Must be possible to do it much better (less communication).

        elem::Matrix<T> A0(DA.Height(), DA.Width(), DA.Height());
        const elem::Matrix<T> &A_local = DA.LockedMatrix();
        elem::Zero(A0);
        for(int j = 0; j < A_local.Width(); j++)
            for(int i = 0; i < A_local.Height(); i++)
                A0.Set(DA.ColShift() + i * DA.ColStride(), j,
                    A_local.Get(i, j));

        boost::mpi::reduce (comm,
            A0.LockedBuffer(),
            A0.MemorySize(),
            A.Buffer(),
            std::plus<T>(),
            0);
    } else {
        std::cout << "NOT SUPPORTED YET!" << std::endl;
    }
}

template<typename T, elem::Distribution ColDist>
void collect_dist_matrix(boost::mpi::communicator& comm, bool here,
    const elem::DistMatrix<T, elem::STAR, ColDist> &DA,
    elem::Matrix<T> &A) {

    if (ColDist == elem::VR || ColDist == elem::VC) {
        // TODO this is probably the most laziest way to do it.
        //      Must be possible to do it much better (less communication).

        elem::Matrix<T> A0(DA.Height(), DA.Width(), DA.Height());
        const elem::Matrix<T> &A_local = DA.LockedMatrix();
        elem::Zero(A0);
        for(int j = 0; j < A_local.Width(); j++)
            for(int i = 0; i < A_local.Height(); i++)
                A0.Set(i, DA.RowShift() + j * DA.RowStride(),
                    A_local.Get(i, j));

        boost::mpi::reduce (comm,
            A0.LockedBuffer(),
            A0.MemorySize(),
            A.Buffer(),
            std::plus<T>(),
            0);
    } else {
        std::cout << "NOT SUPPORTED YET!" << std::endl;
    }
}

} // namespace sketch
} // namespace skylark

#endif
