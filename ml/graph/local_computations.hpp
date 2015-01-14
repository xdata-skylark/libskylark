#ifndef SKYLARK_LOCAL_COMPUTATIONS_HPP
#define SKYLARK_LOCAL_COMPUTATIONS_HPP

#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace skylark { namespace ml {

template<typename GraphType, typename T>
void LocalGraphDiffusion(const GraphType& G,
    const base::sparse_matrix_t<T>& s, base::sparse_matrix_t<T>& y,
    El::Matrix<T> &x, double alpha, double gamma, double epsilon) {

    // TODO verify one column in s.

    const int *seeds = s.indices();
    const double *svalues = s.locked_values();
    int nseeds = s.nonzeros();

    // TODO set N in a better way. It depends on gamma. For gamma=5, N=15
    //      is more than enough, but for gamma=100 it is not (N=50 works).
    int N = 16;

    const double pi = boost::math::constants::pi<double>();
    double LC = 1 + (2 / pi) * log(N);
    double C = (alpha < 1) ?
        (1-alpha) * epsilon / ((1 - exp((alpha - 1) * gamma)) * LC) :
        epsilon / (gamma * LC);

    // Setup matrices associated with Chebyshev spectral diff
    // We cache them to keep costs low (can be crucial for
    // when finding the cluster is very fast).
    static std::unordered_map<int, El::Matrix<T>*> Zmap;
    static std::unordered_map<int, El::Matrix<T>*> Dmap;

    El::Matrix<T> *Z, *D;
    if (Zmap.count(N)) {
        D = Dmap[N];
        Z = Zmap[N];
    } else {
        El::Matrix<T> DO, D1;
        nla::ChebyshevDiffMatrix(N, DO, x, 0, gamma);
        for(int i = 0; i < N; i++)
            DO.Set(i, i, DO.Get(i, i) + 1.0);
        base::ColumnView(D1, DO, 0, N - 1);

        D = new El::Matrix<T>(N, N - 1);
        *D = D1;
        Dmap[N] = D;

        Z = new El::Matrix<T>(N, N - 1);
        *Z = D1;
        El::Pseudoinverse(*Z);
        Zmap[N] = Z;
    }

    // Initialize non-zero functions.
    // TODO: boost or stl?
    std::unordered_map<int, El::Matrix<T>*> yf;
    for (int i = 0; i < nseeds; i++) {
        El::Matrix<T> *f = new El::Matrix<T>(N+1, 1);
        for(int j = 0; j < N; j++)
            f->Set(j, 0, svalues != nullptr ? svalues[i] : 1.0);
        yf[seeds[i]] = f;
    }

    // Compute residual for non-zeros. Put in queue if above threshold.
    typedef std::pair<bool, El::Matrix<T>*> respair_t;
    std::unordered_map<int, respair_t> res;
    std::queue<int> violating;

    // First do seeds
    for (int i = 0; i < nseeds; i++) {
        int node = seeds[i];

        El::Matrix<T> *r = new El::Matrix<T>(N, 1);
        *r = *yf[node];
        El::Scale(-alpha, *r);

        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            int odeg = G.degree(onode);
            if (yf.count(onode))
                El::Axpy(alpha / odeg, *yf[onode], *r);
        }

        bool inq = El::InfinityNorm(*r) > C * deg;
        res[node] = respair_t(inq, r);
        if (inq)
            violating.push(node);
    }

    // Now go over adjanct nodes
    for (int i = 0; i < nseeds; i++) {
        int seed = seeds[i];

        int sdeg = G.degree(seed);
        const int *sadjnodes = G.adjanct(seed);
        for(int j = 0; j < sdeg; j++) {
            int node = sadjnodes[j];
            if (res.count(node))
                continue;

            El::Matrix<T> *r = new El::Matrix<T>(N, 1);
            if (yf.count(node)) {
                *r = *yf[node];
                El::Scale(-alpha, *r);
            } else
                for(int j = 0; j < N; j++)
                    r->Set(j, 0, 0.0);

            int deg = G.degree(node);
            const int *adjnodes = G.adjanct(node);
            for (int l = 0; l < deg; l++) {
                int onode = adjnodes[l];
                int odeg = G.degree(onode);
                if (yf.count(onode))
                    El::Axpy(alpha / odeg, *yf[onode], *r);
            }

            bool inq = El::InfinityNorm(*r) > C * deg;
            res[node] = respair_t(inq, r);
            if (inq)
                violating.push(node);
        }
    }

    El::Matrix<T> dy(N - 1, 1);
    El::Matrix<T> y1, r1;
    while(!violating.empty()) {
        int node = violating.front();
        violating.pop();

        // If not in yf then it is the zero function
        if (yf.count(node) == 0) {
            El::Matrix<T> *ynew = new El::Matrix<T>(N, 1);
            El::Zero(*ynew);
            yf[node] = ynew;
        }

        // Solve locally, and update yf[node].
        respair_t& rpair = res[node];
        El::Matrix<T>& r = *(rpair.second);
        El::Gemv(El::NORMAL, 1.0, *Z, r, 0.0, dy);
        El::View(y1, *yf[node], 0, 0, N - 1, 1);
        El::Axpy(1.0, dy, y1);
        El::Gemv(El::NORMAL, -1.0, *D, dy, 1.0, r);
        rpair.first = false;

        // Update residuals
        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            if (res.count(onode) == 0) {
                El::Matrix<T> *rnew = new El::Matrix<T>(N, 1);
                El::Zero(*rnew);
                res[onode] = respair_t(false, rnew);
            }
            respair_t& rpair1 = res[onode];
            El::View(r1, *(rpair1.second), 0, 0, N - 1, 1);
            El::Axpy(alpha/deg, dy, r1);

            // No need to check if already in queue.
            int odeg = G.degree(onode);
            if (!rpair1.first &&
                El::InfinityNorm(*(rpair1.second)) > C * odeg) {
                rpair1.first = true;
                violating.push(onode);
            }
        }
    }

    // Yank values to y, freeing yf in the process.
    // (we yank at all N time points)
    int nnzcol = yf.size();
    int *yindptr = new int[N + 1];
    for(int i = 0; i < N + 1; i++)
        yindptr[i] = i * nnzcol;
    int *yindices = new int[nnzcol * N];
    double *yvalues = new double[nnzcol * N];

    int idx = 0;
    for(auto it = yf.begin(); it != yf.end(); it++) {
        yindices[idx] = it->first;
        for(int i = 0; i < N; i++)
            yvalues[idx + i * nnzcol] = it->second->Get(i, 0);
        delete it->second;
        idx++;
    }
    y.attach(yindptr, yindices, yvalues, yf.size(), G.num_vertices(),
        N, true);

    // Free res
    for(auto it = res.begin(); it != res.end(); it++)
        delete it->second.second;
}

template<typename GraphType>
double FindLocalCluster(const GraphType& G,
    const std::vector<int>& seeds, std::vector<int>& cluster,
    double alpha, double gamma, double epsilon, bool recursive = true) {

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
        LocalGraphDiffusion(G, s, y, x, alpha, gamma, epsilon);

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
