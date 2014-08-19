#include <iostream>

#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

#include <H5Cpp.h>


#include <unordered_map>
#include <queue>

namespace skylark { namespace ml {
template<typename T>
void LocalGraphDiffusion(const base::sparse_matrix_t<T>& A,
    const base::sparse_matrix_t<T>& s, base::sparse_matrix_t<T>& y,
    double alpha, double gamma, double epsilon) {

    // TODO verify one column in s.
    const int *indptr = A.indptr();
    const int *indices = A.indices();

    const int *seeds = s.indices();
    const double *svalues = s.locked_values();
    int nseeds = s.nonzeros();

    // TODO set N in a better way
    int N = 15;

    const double pi = boost::math::constants::pi<double>();
    double LC = 1 + (2 / pi) * log(N);
    double C = (alpha < 1) ?
        (1-alpha) * epsilon / ((1 - exp((alpha - 1) * gamma)) * LC) :
        epsilon / (gamma * LC);

    // Setup matrices associated with Chebyshev spectral diff
    elem::Matrix<T> DO, ct_N, D1, Z;
    nla::ChebyshevDiffMatrix(N, DO, 0, gamma);
    for(int i = 0; i <= N; i++)
        DO.Set(i, i, DO.Get(i, i) + 1.0);
    base::ColumnView(ct_N, DO, N, 1);
    base::ColumnView(D1, DO, 0, N);
    Z = D1;
    elem::Pseudoinverse(Z);

    // Initlize non-zero functions.
    // TODO: boost or stl?
    std::unordered_map<int, elem::Matrix<T>*> yf;
    for (int i = 0; i < nseeds; i++) {
        elem::Matrix<T> *f = new elem::Matrix<T>(N+1, 1);
        for(int j = 0; j <= N; j++)
            f->Set(j, 0, svalues[i]);
        yf[seeds[i]] = f;
    }

    // Compute residual for non-zeros. Put in queue if above threshold.
    typedef std::pair<bool, elem::Matrix<T>*> respair_t;
    std::unordered_map<int, respair_t> res;
    std::queue<int> violating;

    // First do seeds
    for (int i = 0; i < nseeds; i++) {
        int node = seeds[i];

        elem::Matrix<T> *r = new elem::Matrix<T>(N+1, 1);
        *r = *yf[node];
        elem::Scal(-alpha, *r);

        for (int l = indptr[node]; l < indptr[node+1]; l++) {
            int onode = indices[l];
            int deg = indptr[onode + 1] - indptr[onode];
            if (yf.count(onode))
                elem::Axpy(alpha / deg, *yf[onode], *r);
        }

        int deg = indptr[node + 1] - indptr[node];
        bool inq = elem::InfinityNorm(*r) > C * deg;
        res[node] = respair_t(inq, r);
        if (inq)
            violating.push(node);
    }

    // Now go over adjanct nodes
    for (int i = 0; i < nseeds; i++) {
        int seed = seeds[i];

        for(int j = indptr[seed]; j < indptr[seed+1]; j++) {
            int node = indices[j];
            if (res.count(node))
                continue;

            elem::Matrix<T> *r = new elem::Matrix<T>(N+1, 1);
            if (yf.count(node)) {
                *r = *yf[node];
                elem::Scal(-alpha, *r);
            } else
                for(int j = 0; j <= N; j++)
                    r->Set(j, 0, 0.0);

            for (int l = indptr[node]; l < indptr[node+1]; l++) {
                int onode = indices[l];
                int deg = indptr[onode + 1] - indptr[onode];
                if (yf.count(onode))
                    elem::Axpy(alpha / deg, *yf[onode], *r);
            }

            int deg = indptr[node + 1] - indptr[node];
            bool inq = elem::InfinityNorm(*r) > C * deg;
            res[node] = respair_t(inq, r);
            if (inq)
                violating.push(node);
        }
    }

    int it = 1;
    elem::Matrix<T> dy(N, 1);
    elem::Matrix<T> y1, r1;
    while(!violating.empty()) {
        int node = violating.front();
        violating.pop();

        // If not in yf then it is the zero function
        if (yf.count(node) == 0) {
            elem::Matrix<T> *ynew = new elem::Matrix<T>(N+1, 1);
            elem::MakeZeros(*ynew);
            yf[node] = ynew;
        }

        // Solve locally, and update yf[node].
        respair_t& rpair = res[node];
        elem::Matrix<T>& r = *(rpair.second);
        elem::Gemv(elem::NORMAL, 1.0, Z, r, 0.0, dy);
        elem::View(y1, *yf[node], 0, 0, N, 1);
        elem::Axpy(1.0, dy, y1);
        elem::Gemv(elem::NORMAL, -1.0, D1, dy, 1.0, r);
        rpair.first = false;

        // Update residuals
        int deg = indptr[node + 1] - indptr[node];
        for (int l = indptr[node]; l < indptr[node+1]; l++) {
            int onode = indices[l];
            if (res.count(onode) == 0) {
                elem::Matrix<T> *rnew = new elem::Matrix<T>(N+1, 1);
                elem::MakeZeros(*rnew);
                res[onode] = respair_t(false, rnew);
            }
            respair_t& rpair1 = res[onode];
            elem::View(r1, *(rpair1.second), 0, 0, N, 1);
            elem::Axpy(alpha/deg, dy, r1);

            // No need to check if already in queue.
            int odeg = indptr[onode + 1] - indptr[onode];
            if (!rpair1.first &&
                elem::InfinityNorm(*(rpair1.second)) > C * odeg) {
                rpair1.first = true;
                violating.push(onode);
            }
        }

        it = it+1;
    }

    // Free memory
    for(typename std::unordered_map<int, elem::Matrix<T>*>::iterator it =
            yf.begin(); it != yf.end(); it++)
        delete it->second;
    for(typename std::unordered_map<int, respair_t>::iterator it =
            res.begin(); it != res.end(); it++)
        delete it->second.second;
}

} }

namespace bmpi =  boost::mpi;
namespace skybase = skylark::base;
namespace skysketch =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyml = skylark::ml;
namespace skyutil = skylark::utility;

int main(int argc, char** argv) {

    elem::Initialize(argc, argv);
    skybase::context_t context(23234);

    skybase::sparse_matrix_t<double> A;
    elem::Matrix<double> b;

    boost::mpi::timer timer;

    // Load A and b from HDF5 file
    std::cout << "Reading the adjacency matrix... ";
    std::cout.flush();
    timer.restart();
    H5::H5File in(argv[1], H5F_ACC_RDONLY);
    skyutil::io::ReadHDF5(in, "A", A);
    in.close();
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    // TODO get these as parameters
    int seed = 19043 - 1;
    double gamma = 5;
    double epsilon = 0.001;
    double alpha = 0.8;

    // Create seed vector.
    int indptr[2] = {0, 1};
    int indices[1] = {seed};
    double values[1] = {1.0};
    skybase::sparse_matrix_t<double> s;
    s.attach(indptr, indices, values, 1, A.height(), 1);

    skybase::sparse_matrix_t<double> y;
    timer.restart();
    skyml::LocalGraphDiffusion(A, s, y, alpha, gamma, epsilon);
    std::cout <<"Took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    return 0;
}
