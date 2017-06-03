#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <El.hpp>

int max(El::Matrix<double> Y) {
    int k =  (int) *std::max_element(Y.Buffer(), Y.Buffer() + Y.Height());
    return k;
}

int min(El::Matrix<double> Y) {
    int k =  (int) *std::min_element(Y.Buffer(), Y.Buffer() + Y.Height());
    return k;
}

template<class LabelType>
int GetNumTargets(const boost::mpi::communicator &comm, LabelType& Y) {
    int k = 0;
    int kmax = max(Y);
    int kmin = min(Y);
    int targets = 0;

    boost::mpi::all_reduce(comm, kmin, k, boost::mpi::minimum<int>());
    if (k==-1) {
    	targets = 1;
    }
    else {
    	boost::mpi::all_reduce(comm, kmax, k, boost::mpi::maximum<int>());
    	targets = k+1;
    }

    return targets;
}

#endif
