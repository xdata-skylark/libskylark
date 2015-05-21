#ifndef SKYLARK_RLSC_HPP
#define SKYLARK_RLSC_HPP

namespace skylark { namespace ml {

template<typename T, typename R, typename KernelType>
void KernelRLSC(base::direction_t direction, const KernelType &k, 
    const El::DistMatrix<T> &X, const El::DistMatrix<R> &L, T lambda, 
    El::DistMatrix<T> &A, std::vector<R> &rcoding) {

    // TODO: Temporary!
    boost::mpi::communicator world;
    int rank = world.rank();

    boost::mpi::timer timer;

    // Form right hand side
    if (rank == 0) {
        std::cout << "Dummy coding... ";
        std::cout.flush();
        timer.restart();
    }

    El::DistMatrix<T> Y;
    std::unordered_map<R, El::Int> coding;
    skylark::ml::DummyCoding(El::NORMAL, Y, L, coding, rcoding);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    // Solve
    if (rank == 0) {
        std::cout << "Solving... " << std::endl;
        timer.restart();
    }

    skylark::ml::KernelRidge(skylark::base::COLUMNS, k, X, Y, T(lambda), A);

    if (rank == 0)
        std::cout <<"Solve took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

}

template<typename T, typename R, typename KernelType>
void FasterKernelRLSC(base::direction_t direction, const KernelType &k,
    const El::DistMatrix<T> &X, const El::DistMatrix<R> &L, T lambda,
    El::DistMatrix<T> &A, std::vector<R> &rcoding,
    El::Int s, base::context_t &context) {

    // TODO: Temporary!
    boost::mpi::communicator world;
    int rank = world.rank();

    boost::mpi::timer timer;

    // Form right hand side
    if (rank == 0) {
        std::cout << "Dummy coding... ";
        std::cout.flush();
        timer.restart();
    }

    El::DistMatrix<T> Y;
    std::unordered_map<R, El::Int> coding;
    skylark::ml::DummyCoding(El::NORMAL, Y, L, coding, rcoding);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    // Solve
    if (rank == 0) {
        std::cout << "Solving... " << std::endl;
        timer.restart();
    }

    skylark::ml::FasterKernelRidge(direction, k, X, Y,
        T(lambda), A, s, context);

    if (rank == 0)
        std::cout <<"Solve took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

}


} } // namespace skylark::ml

#endif
