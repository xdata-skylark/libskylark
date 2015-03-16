#include <iostream>

#include <El.hpp>
#include <skylark.hpp>

const int m = 5000;
const int n = 100;
const int k = 10;

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    boost::mpi::communicator world;
    int rank = world.rank();

    skylark::base::context_t context(38734);

    /** Generate matrices U, S, V*/
    El::DistMatrix<double> U;
    skylark::base::UniformMatrix(U, m, n, context);
    skylark::base::qr::ExplicitUnitary(U);

    El::DistMatrix<double> V;
    skylark::base::UniformMatrix(V, n, n, context);
    skylark::base::qr::ExplicitUnitary(V);

    El::DistMatrix<double> S(n, 1);
    for(int i = 0; i < n; i++) S.Set(i, 0, exp(-i) * 100);

    /* Compute A = U * S * V^T */
    El::DistMatrix<double> VS = V;
    El::DiagonalScale(El::RIGHT, El::NORMAL, S, VS);

    El::DistMatrix<double> A;
    El::Gemm(El::NORMAL, El::ADJOINT, 1.0, U, VS, A);

    /* Compute approximate SVD */
    skylark::nla::approximate_svd_params_t params;
    params.skip_qr = false;
    params.num_iterations = 2;

    El::DistMatrix<double> A1;
    El::Transpose(A, A1);

    El::DistMatrix<double> U1, S1, V1;
    skylark::nla::ApproximateSVD(A1, U1, S1, V1, k, context, params);

    for(int i = 0; i < k; i++) {
        std::cout << "TRUE: " << S.Get(i, 0) << "\tAPPROX: " << S1.Get(i, 0)
                  << "\tRelative error: "
                  << std::abs(S.Get(i, 0) - S1.Get(i, 0)) / S.Get(i, 0)
                  << std::endl;

    }

    El::Finalize();
    return 0;
}
