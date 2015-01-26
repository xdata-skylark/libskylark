#ifndef SKYLARK_LOCAL_COMPUTATIONS_HPP
#define SKYLARK_LOCAL_COMPUTATIONS_HPP

#include <boost/math/special_functions/bessel.hpp>

#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace skylark { namespace ml {

template<typename GraphType, typename T>
void LocalGraphDiffusion(const GraphType& G,
    const base::sparse_matrix_t<T>& s, base::sparse_matrix_t<T>& y,
    El::Matrix<T> &x, double alpha, double gamma, double epsilon, int NX = 4) {

    if (s.width() != 1)
        SKYLARK_THROW_EXCEPTION (
            base::invalid_parameters()
               << base::error_msg("input should be a vector") );

    const int *seeds = s.indices();
    const double *svalues = s.locked_values();
    int nseeds = s.nonzeros();

    // Find minimum N, caching it since it involves costly computations of
    // Bessel functions.
    static std::unordered_map<std::pair<double, double>, int,
                              utility::pair_hasher_t> Nmap;
    auto epsgamma = std::make_pair(epsilon, gamma);
    int minN;
    const double pi = boost::math::constants::pi<double>();
    if (Nmap.count(epsgamma))
        minN = Nmap[epsgamma];
    else {
        minN = 10;
        double C = 20.0 * std::sqrt(minN) * std::exp(-gamma/2);
        while (C * boost::math::cyl_bessel_i(minN, gamma) * pow(0.8, minN) >
            epsilon / (gamma * (1 + (2 / pi) * log(minN - 1))))
            minN++;
        Nmap[epsgamma] = minN;
    }

    int N = (minN / NX + 1) * NX;   // This verifies that N is a multiple of NX.
    int NR = N / NX;

    double LC = 1 + (2 / pi) * log(N - 1);
    double C = (alpha < 1) ?
        (1-alpha) * epsilon / ((1 - exp((alpha - 1) * gamma)) * LC) :
        epsilon / (gamma * LC);

    // Setup matrices associated with Chebyshev spectral diff
    // We cache them to keep costs low (can be crucial for
    // when finding the cluster is very fast).
    static std::unordered_map<std::pair<int, double>, El::Matrix<T>*,
                              utility::pair_hasher_t> DZmap;

    El::Matrix<T> *DZ;
    auto ngamma = std::make_pair(N, gamma);
    if (DZmap.count(ngamma)) {
        DZ = DZmap[ngamma];
        nla::ChebyshevPoints(N, x, 0, gamma);
    } else {
        El::Matrix<T> DO, D1, x1;
        nla::ChebyshevDiffMatrix(N, DO, x1, 0, gamma);
        for(int i = 0; i < N; i++)
            DO.Set(i, i, DO.Get(i, i) + 1.0);
        base::ColumnView(D1, DO, 0, N - 1);

        El::Matrix<T> Z(N, N - 1);
        Z = D1;
        El::Pseudoinverse(Z);

        El::Matrix<T> D(N, N);
        El::Gemm(El::NORMAL, El::NORMAL, 1.0, D1, Z, 0.0, D);

        DZ = new El::Matrix<T>(2 * N - 1, N);

        El::Matrix<T> V;
        base::RowView(V, *DZ, 0, N);
        V = D;
        base::RowView(V, *DZ, N, N-1);
        V = Z;

        DZmap[ngamma] = DZ;

        x.Resize(NX, 1);
        for(int i = 0; i < NX; i++)
            x.Set(i, 0, x1.Get(i * NR, 0));
    }

    typedef std::pair<bool, El::Matrix<T>*> rypair_t;
    std::unordered_map<int, rypair_t> rymap;
    std::queue<int> violating;

    // Initialize non-zero functions, and their residual, which is not
    // fully computed yet (but we know that needs to be inserted into the queue).
    for (int i = 0; i < nseeds; i++) {
        int node = seeds[i];

        El::Matrix<T> *ry = new El::Matrix<T>(N + NX, 1);
        for(int j = 0; j < N; j++)
            ry->Set(j, 0, svalues != nullptr ? -alpha * svalues[i] : -alpha);
        for(int j = 0; j < NX; j++)
            ry->Set(N + j, 0, svalues != nullptr ? svalues[i] : 1.0);

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
                El::Matrix<T> *ry = new El::Matrix<T>(N + NX, 1);
                El::Zero(*ry);
                rymap[onode] = rypair_t(false, ry);
            }
        }
    }

    // Update the residual based on seeds
    for (int i = 0; i < nseeds; i++) {
        int node = seeds[i];

        El::Matrix<T> *ry = rymap[node].second;

        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        double v = alpha * ry->Get(N,0) / deg;
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            int odeg = G.degree(onode);

            rypair_t& ryopair = rymap[onode];
            T *robuf = ryopair.second->Buffer();
            bool inq = false;
            for(int j = 0; j < N; j++) {
                robuf[j] += v;
                inq = inq || (std::abs(robuf[j]) > C * odeg);
            }
            inq = inq || (std::abs(robuf[N-1]) > C * odeg);
            if (!ryopair.first && inq) {
                violating.push(onode);
                ryopair.first = true;
            }
        }
    }

    // Initialize r and y for nodes that are adjanct to seeds
    El::Matrix<T> dry(2 * N - 1, 1);
    El::Matrix<T> dr = base::RowView(dry, 0, N);
    T *dybuf = dry.Buffer() + N;
    El::Matrix<T> r;
    while(!violating.empty()) {
        int node = violating.front();
        violating.pop();

        // Solve locally, and update rymap[node].
        rypair_t& rpair = rymap[node];
        El::Matrix<T>& ry = *(rpair.second);

        // Compute change in sample values, and r value (of this node)
        base::RowView(r, ry, 0, N);
        El::Gemv(El::NORMAL, 1.0, *DZ, r, 0.0, dry);
        El::Axpy(-1.0, dr, r);
        T *ybuf = ry.Buffer() + N;
        for(int i = 0; i < NX; i++)
            ybuf[i] += dybuf[i * NR];

        rpair.first = false;

        // Update residuals
        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            int odeg = G.degree(onode);

            // Add it to rymap, if not already there.
            if (rymap.count(onode) == 0) {
                El::Matrix<T> *rynew = new El::Matrix<T>(N + NX, 1);
                El::Zero(*rynew);
                rymap[onode] = rypair_t(false, rynew);
            }

            rypair_t& ryopair = rymap[onode];
            bool inq = false;
            T *robuf = ryopair.second->Buffer();
            for(int i = 0; i < N - 1; i++) {
                robuf[i] += alpha * dybuf[i] / deg;
                inq = inq || (std::abs(robuf[i]) > C * odeg);
            }
            inq = inq || (std::abs(robuf[N - 1]) > C * odeg);
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
            yvalues[idx + i * nnzcol] = it->second.second->Get(N + i, 0);
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
