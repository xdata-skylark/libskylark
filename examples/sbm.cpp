#include <skylark.hpp>

namespace skyml = skylark::ml;

int main(int argc, char** argv) {
    El::DistMatrix<double> A;
    El::Uniform(A, 32, 16);

    constexpr unsigned d = 8;
    constexpr unsigned k = 8;
    constexpr unsigned powerits = 1;
    skyml::kmeans_t<double> ret =
        skyml::sketched_sbm_clusters<double>(A, d, k, powerits);

    ret.print();
}
