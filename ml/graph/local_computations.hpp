#ifndef SKYLARK_LOCAL_COMPUTATIONS_HPP
#define SKYLARK_LOCAL_COMPUTATIONS_HPP

#include <boost/math/special_functions/bessel.hpp>

#include <unordered_map>
#include <unordered_set>
#include <queue>

extern "C" {

void EL_BLAS(sgemv)(const char*, const El::Int *, const El::Int *,
    const float *, const float *, const El::Int *,
    const float *, const El::Int *, const float *, float *, const El::Int *);

void EL_BLAS(dgemv)(const char*, const El::Int *, const El::Int *,
    const double *, const double *, const El::Int *,
    const double *, const El::Int *, const double *, double *, const El::Int *);

}

namespace skylark { namespace ml {

template<typename GraphType, typename T>
void LocalGraphDiffusion(const GraphType& G,
    const base::sparse_matrix_t<T>& s, base::sparse_matrix_t<T>& y,
    El::Matrix<T> &x, double alpha, double gamma, double epsilon, int NX = 4) {

    if (s.width() != 1)
        SKYLARK_THROW_EXCEPTION (
            base::invalid_parameters()
               << base::error_msg("input should be a vector") );

    if (!El::Initialized())
        SKYLARK_THROW_EXCEPTION (
            base::skylark_exception()
               << base::error_msg("Elemental was not initialized") );

    const int *seeds = s.indices();
    const double *svalues = s.locked_values();
    int nseeds = s.nonzeros();

    // Find minimum N, caching it since it involves costly computations of
    // Bessel functions.
    static std::unordered_map<std::pair<double, double>, int,
                              utility::pair_hasher_t> Nmap;
    auto epsgamma = std::make_pair(epsilon, gamma);
    const double pi = boost::math::constants::pi<double>();
    if (Nmap.count(epsgamma) == 0) {
        int minN = 10;
        double C = 20.0 * std::sqrt(minN) * std::exp(-gamma/2);
        while (C * boost::math::cyl_bessel_i(minN, gamma) * pow(0.8, minN) >
            epsilon / (gamma * (1 + (2 / pi) * log(minN - 1))))
            minN++;
        Nmap[epsgamma] = minN;
    }
    int minN = Nmap[epsgamma];

    // N is taken to be the minimum multiple of NX that is bigger or equal
    // to minN.
    int N = minN % NX == 0 ? minN : (minN / NX + 1) * NX;
    int NR = N / NX;

    // Setup matrices associated with Chebyshev spectral diff
    // We cache them to keep costs low (can be crucial for
    // when finding the cluster is very fast).
    static std::unordered_map<std::pair<int, double>, El::Matrix<T>*,
                              utility::pair_hasher_t> Dmap;

    auto ngamma = std::make_pair(N, gamma);
    if (Dmap.count(ngamma) == 0) {
        El::Matrix<T> *D = new El::Matrix<T>(N, N);
        Dmap[ngamma] = D;

        El::Matrix<T> D0;
        nla::ChebyshevDiffMatrix(N, D0, x, 0, gamma);
        for(int i = 0; i < N; i++)
            D0.Set(i, i, D0.Get(i, i) + 1.0);

        El::Matrix<T> R(N, N);
        El::qr::Explicit(D0, R);

        for(int j = 0; j < N; j++)
            D->Set(N-1, j, D0.Get(j, N-1));

        El::Matrix<T> Q1, R1;
        base::ColumnView(Q1, D0, 0, N - 1);
        El::View(R1, R, 0, 0, N-1, N-1);

        El::Pseudoinverse(R1);

        El::Matrix<T> DU;
        base::RowView(DU, *D, 0, N-1);
        El::Gemm(El::NORMAL, El::TRANSPOSE, 1.0, R1, Q1, 0.0, DU);
    }

    const El::Matrix<T> *D_ = Dmap[ngamma];

    // TODO the following line is not correct.
    nla::ChebyshevPoints(N, x, 0, gamma);

    // Constants for convergence.
    double LC = 1 + (2 / pi) * log(N - 1);
    double C = (alpha < 1) ?
        (1-alpha) * epsilon / ((1 - exp((alpha - 1) * gamma)) * LC) :
        epsilon / (gamma * LC);

    // From now on, do not use Elemental to avoid overheads.
    const T *D = D_->LockedBuffer();
    const T *u = D_->LockedBuffer() + N - 1;

    typedef std::pair<bool, T*> rypair_t;
    std::unordered_map<int, rypair_t> rymap;
    std::queue<int> violating;

    // Initialize non-zero functions, and their residual, which is not
    // fully computed yet (but we know that needs to be inserted into the queue).
    for (int i = 0; i < nseeds; i++) {
        T *ry = new T[N + NX];
        std::fill(ry, ry + N, svalues != nullptr ? -alpha * svalues[i] : -alpha);
        std::fill(ry + N, ry + N + NX, svalues != nullptr ? svalues[i] : 1.0);

        int node = seeds[i];
        rymap[node] = rypair_t(true, ry);
        violating.push(node);
    }

    // Initialize to just zero for all nodes adjanct to seeds, that
    // are not seeds themselves. Residual is not fully computed yet.
    for (int i = 0; i < nseeds; i++) {
        const int *adjnodes = G.adjanct(seeds[i]);
        for (int l = 0; l < G.degree(seeds[i]); l++) {
            int onode = adjnodes[l];
            if (rymap.count(onode) == 0) {
                T *ry = new T[N + NX];
                std::fill(ry, ry + N + NX, 0);
                rymap[onode] = rypair_t(false, ry);
            }
        }
    }

    // Update the residual based on seeds
    for (int i = 0; i < nseeds; i++) {
        int node = seeds[i];

        T *ry = rymap[node].second;

        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        double v = alpha * ry[N] / deg;
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            int odeg = G.degree(onode);

            rypair_t& ryopair = rymap[onode];
            T *ro = ryopair.second;
            bool inq = false;
            double B = C * odeg;
            for(int j = 0; j < N; j++) {
                ro[j] += v;
                inq = inq || (std::abs(ro[j]) > B);
            }
            if (!ryopair.first && inq) {
                violating.push(onode);
                ryopair.first = true;
            }
        }
    }

    // Main loop
    T dyp[N];
    while(!violating.empty()) {
        int node = violating.front();
        violating.pop();

        // Solve locally, and update rymap[node].
        rypair_t& rpair = rymap[node];
        T *ry = rpair.second;

        // Compute correction to y, and the new residual.
        // TODO double or single
        T done = 1.0, dzero = 0.0;
        El::Int ione = 1;
        if (std::is_same<T, float>::value)
            EL_BLAS(sgemv)("Normal", &N, &N, (float *)&done, (float *)D, &N,
                (float *)ry, &ione, (float *)&dzero, (float *)dyp, &ione);
        else
            EL_BLAS(dgemv)("Normal", &N, &N, (double *)&done, (double *)D, &N,
                (double *)ry, &ione, (double *)&dzero, (double *)dyp, &ione);
        for(int i = 0; i < NX; i++)
            ry[N + i] += dyp[i * NR];
        T v = dyp[N-1];
        for(int i = 0; i < N; i++)
            ry[i] = v * u[i * N];

        // No longer in queue.
        rpair.first = false;

        // Update residuals
        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            int odeg = G.degree(onode);

            // Add it to rymap, if not already there.
            if (rymap.count(onode) == 0) {
                T *rynew = new T[N + NX];
                std::fill(rynew, rynew + N + NX, 0);
                rymap[onode] = rypair_t(false, rynew);
            }

            rypair_t& ryopair = rymap[onode];
            bool inq = false;
            T *ryo = ryopair.second;
            double c = alpha / deg;
            double B = C * odeg;
            for(int i = 0; i < N - 1; i++) {
                ryo[i] += c *  dyp[i];
                inq = inq || (std::abs(ryo[i]) > B);
            }
            inq = inq || (std::abs(ryo[N - 1]) > B);
            if (!ryopair.first && inq) {
                violating.push(onode);
                ryopair.first = true;
            }
        }
    }

    // Yank values to y, freeing ry in the process.
    // (we yank at all N time points)
    int nnzcol = rymap.size();
    int *yindptr = new int[NX + 1];
    for(int i = 0; i < NX + 1; i++)
        yindptr[i] = i * nnzcol;
    int *yindices = new int[nnzcol * NX];
    double *yvalues = new double[nnzcol * NX];

    int idx = 0;
    for(auto it = rymap.begin(); it != rymap.end(); it++) {
        for(int i = 0; i < NX; i++) {
            yindices[idx + i * nnzcol] = it->first;
            yvalues[idx + i * nnzcol] = it->second.second[N + i];
        }
        delete it->second.second;
        idx++;
    }
    y.attach(yindptr, yindices, yvalues, rymap.size(), G.num_vertices(),
        NX, true);
}

template<typename GraphType>
double FindLocalCluster(const GraphType& G,
    const std::vector<int>& seeds, std::vector<int>& cluster,
    double alpha, double gamma, double epsilon, int NX = 4,
    bool recursive = true) {

    double currentcond = -1;
    cluster = seeds;
    bool improve;
    El::Matrix<double> x;

    do {
        // Create seed vector.
        int sindptr[2] = {0, static_cast<int>(cluster.size())};
        base::sparse_matrix_t<double> s;
        s.attach(sindptr, cluster.data(), nullptr, cluster.size(),
            G.num_vertices(), 1);

        // Run the diffusion
        base::sparse_matrix_t<double> y;
        LocalGraphDiffusion(G, s, y, x, alpha, gamma, epsilon, NX);

        // Go over the y output at the different time samples,
        // find the best prefix and if better conductance, store it.
        improve = false;
        const int *indptr = y.indptr();
        const int *indices = y.indices();
        const double *values = y.locked_values();
        for (int t = 0; t < y.width(); t++) {
            // Sort (descending) the non-zero components based on their normalized
            // y values (normalized by degree).
            std::vector<std::pair<double, int> > vals(indptr[t + 1] - indptr[t]);
            const double *yvalues = values + indptr[t];
            const int *yindices = indices + indptr[t];
            for(int i = 0; i < y.nonzeros(); i++) {
                int idx = yindices[i];
                double val = - yvalues[i] / G.degree(idx);
                vals[i] = std::pair<double, int>(val, idx);
            }
            std::sort(vals.begin(), vals.end());

            // Find the best prefix
            int volS = 0, cutS = 0;
            double bestcond = 1.0;
            int bestprefix = 0;
            int Gvol = G.num_edges();
            std::unordered_set<int> currentset;
            for (int i = 0; i < vals.size(); i++) {
                int node = vals[i].second;
                int deg = G.degree(node);
                const int *adjnodes = G.adjanct(node);
                volS += deg;
                for(int l = 0; l < deg; l++) {
                    int onode = adjnodes[l];
                    if (currentset.count(onode))
                        cutS--;
                    else
                        cutS++;
                }

                double condS =
                    static_cast<double>(cutS) / std::min(volS, Gvol - volS);
                if (condS < bestcond) {
                    bestcond = condS;
                    bestprefix = i;
                }
                currentset.insert(node);
            }

            if (currentcond == -1 || bestcond < 0.999999 * currentcond) {
                // We have a new best cluster - the best perfix.
                improve = true;
                cluster.clear();
                for(int i = 0; i <= bestprefix; i++)
                    cluster.push_back(vals[i].second);
                currentcond = bestcond;
            }
        }
    } while (recursive && improve);

    return currentcond;
}

} }   // namespace skylark::ml

#endif // SKYLARK_LOCAL_COMPUTATIONS_HPP
