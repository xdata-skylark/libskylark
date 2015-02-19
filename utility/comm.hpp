#ifndef SKYLARK_COMM_HPP
#define SKYLARK_COMM_HPP

#include <El.hpp>

#include "../base/exception.hpp"

namespace skylark {
namespace utility {

template<typename T>
void collect_dist_matrix(boost::mpi::communicator& comm, bool here,
    const El::DistMatrix<T> &DA, El::Matrix<T> &A) {

    // Technically the following should work for any row and col distributions.
    // But it seems to not work well for VR/VC type distributions.
    // And it is probably not the best way to do it for these distributions
    // anyway.

    try {
        El::AxpyInterface<T> interface;
        interface.Attach(El::GLOBAL_TO_LOCAL, DA);
        if (here) {
            El::Zero(A);
            interface.Axpy(1.0, A, 0, 0);
        }
        interface.Detach();
    } catch (std::logic_error e) {
        SKYLARK_THROW_EXCEPTION (base::elemental_exception()
            << base::error_msg(e.what()) );
    }
}

template<typename T, El::Distribution RowDist>
void collect_dist_matrix(boost::mpi::communicator& comm, bool here,
    const El::DistMatrix<T, RowDist, El::STAR> &DA,
    El::Matrix<T> &A) {

    if (RowDist == El::VR || RowDist == El::VC) {
        // TODO this is probably the most laziest way to do it.
        //      Must be possible to do it much better (less communication).

        try {
            El::Matrix<T> A0(DA.Height(), DA.Width(), DA.Height());
            const El::Matrix<T> &A_local = DA.LockedMatrix();
            El::Zero(A0);
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

        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION( base::elemental_exception()
                << base::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION(base::mpi_exception()
                << base::error_msg(e.what()) );
        }

    } else {
        SKYLARK_THROW_EXCEPTION ( base::unsupported_matrix_distribution() );
    }
}

template<typename T, El::Distribution ColDist>
void collect_dist_matrix(boost::mpi::communicator& comm, bool here,
    const El::DistMatrix<T, El::STAR, ColDist> &DA,
    El::Matrix<T> &A) {

    if (ColDist == El::VR || ColDist == El::VC) {
        // TODO this is probably the most laziest way to do it.
        //      Must be possible to do it much better (less communication).

        try {
            El::Matrix<T> A0(DA.Height(), DA.Width(), DA.Height());
            const El::Matrix<T> &A_local = DA.LockedMatrix();
            El::Zero(A0);
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

        } catch (std::logic_error e) {
            SKYLARK_THROW_EXCEPTION (base::elemental_exception()
                << base::error_msg(e.what()) );
        } catch(boost::mpi::exception e) {
            SKYLARK_THROW_EXCEPTION (base::mpi_exception()
                << base::error_msg(e.what()) );
        }

    } else {
        SKYLARK_THROW_EXCEPTION ( base::unsupported_matrix_distribution() );
    }
}

} // namespace sketch
} // namespace skylark

#endif
