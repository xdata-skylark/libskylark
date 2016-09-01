#include <iostream>
#include <El.hpp>
#include <skylark.hpp>
#include <boost/mpi.hpp>


int main(int argc, char* argv[]) {

    // Config parameters
    int seed = 38734;
    std::string fname = "/home/jms/notebooks/data/usps.t";

    // Initialize
    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();

    skylark::base::context_t context(seed);


    cout << "ExpSemiGroup Kernel: " << endl;
    El::DistMatrix<double> X0, X, Y0, Y, K;

    // Read the file
    skylark::utility::io::ReadLIBSVM(fname, X0, Y0, skylark::base::COLUMNS,
            skylark::base::COLUMNS, -1);

    // Does not sample the data
    El::View(X, X0);
    El::View(Y, Y0);

    // Calculate the kernel
    std::shared_ptr<skylark::ml::kernel_t> k_ptr;
    k_ptr.reset(new skylark::ml::expsemigroup_t(X.Height(), 0.01));
    skylark::ml::kernel_container_t k(k_ptr);
    skylark::ml::Gram(skylark::base::COLUMNS, skylark::base::COLUMNS, k, X, X, K);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 5; ++j) {
            cout << K.Get(i,j) << " ";
        }
        cout << endl;
    }


    cout << "Gaussian Kernel: " << endl;

    El::DistMatrix<double> X02, X2, Y02, Y2, K2;

    // Read the file
    skylark::utility::io::ReadLIBSVM(fname, X02, Y02, skylark::base::COLUMNS,
            skylark::base::COLUMNS, -1);

    // Does not sample the data
    El::View(X2, X02);
    El::View(Y2, Y02);
    k_ptr.reset(new skylark::ml::gaussian_t(X2.Height(), 10.0));
    skylark::ml::kernel_container_t k2(k_ptr);
    skylark::ml::Gram(skylark::base::COLUMNS, skylark::base::COLUMNS, k2, X2, X2, K2);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 5; ++j) {
            cout << K2.Get(i,j) << " ";
        }
        cout << endl;
    }

    El::Finalize();
    return 0;
}