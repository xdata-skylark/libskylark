#ifndef SKYLARK_LOCAL_COMPUTATIONS_HPP
#define SKYLARK_LOCAL_COMPUTATIONS_HPP

#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace skylark { namespace ml {

template<typename GraphType, typename T>
void LocalGraphDiffusion(const GraphType& G,
    const base::sparse_matrix_t<T>& s, base::sparse_matrix_t<T>& y,
    double alpha, double gamma, double epsilon) {

    // TODO verify one column in s.

    const int *seeds = s.indices();
    const double *svalues = s.locked_values();
    int nseeds = s.nonzeros();

    // TODO set N in a better way. It depends on gamma. For gamma=5, N=15
    //      is more than enough, but for gamma=100 it is not (N=50 works).
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
            f->Set(j, 0, svalues != nullptr ? svalues[i] : 1.0);
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

        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            int odeg = G.degree(onode);
            if (yf.count(onode))
                elem::Axpy(alpha / odeg, *yf[onode], *r);
        }

        bool inq = elem::InfinityNorm(*r) > C * deg;
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

            elem::Matrix<T> *r = new elem::Matrix<T>(N+1, 1);
            if (yf.count(node)) {
                *r = *yf[node];
                elem::Scal(-alpha, *r);
            } else
                for(int j = 0; j <= N; j++)
                    r->Set(j, 0, 0.0);

            int deg = G.degree(node);
            const int *adjnodes = G.adjanct(node);
            for (int l = 0; l < deg; l++) {
                int onode = adjnodes[l];
                int odeg = G.degree(onode);
                if (yf.count(onode))
                    elem::Axpy(alpha / odeg, *yf[onode], *r);
            }

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
        int deg = G.degree(node);
        const int *adjnodes = G.adjanct(node);
        for (int l = 0; l < deg; l++) {
            int onode = adjnodes[l];
            if (res.count(onode) == 0) {
                elem::Matrix<T> *rnew = new elem::Matrix<T>(N+1, 1);
                elem::MakeZeros(*rnew);
                res[onode] = respair_t(false, rnew);
            }
            respair_t& rpair1 = res[onode];
            elem::View(r1, *(rpair1.second), 0, 0, N, 1);
            elem::Axpy(alpha/deg, dy, r1);

            // No need to check if already in queue.
            int odeg = G.degree(onode);
            if (!rpair1.first &&
                elem::InfinityNorm(*(rpair1.second)) > C * odeg) {
                rpair1.first = true;
                violating.push(onode);
            }
        }

        it = it+1;
    }

    // Yank values to y, freeing yf in the process.
    int *yindptr = new int[2]; yindptr[0] = 0; yindptr[1] = yf.size();
    int *yindices = new int[yf.size()];
    double *yvalues = new double[yf.size()];
    int idx = 0;
    for(auto it = yf.begin(); it != yf.end(); it++) {
        yindices[idx] = it->first;
        yvalues[idx] = it->second->Get(0, 0);
        delete it->second;
        idx++;
    }
    y.attach(yindptr, yindices, yvalues, yf.size(), G.num_vertices(), 1, true);

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

    while(true) {
        // Create seed vector.
        int sindptr[2] = {0, static_cast<int>(cluster.size())};
        base::sparse_matrix_t<double> s;
        s.attach(sindptr, cluster.data(), nullptr, cluster.size(),
            G.num_vertices(), 1);

        // Run the diffusion
        base::sparse_matrix_t<double> y;
        LocalGraphDiffusion(G, s, y, alpha, gamma, epsilon);

        // Sort (descending) the non-zero components based on their normalized
        // y values (normalized by degree).
        std::vector<std::pair<double, int> > vals(y.nonzeros());
        const double *yvalues = y.locked_values();
        const int *yindices = y.indices();
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
            cluster.clear();
            for(int i = 0; i <= bestprefix; i++)
                cluster.push_back(vals[i].second);
            currentcond = bestcond;

            if (!recursive)
                break;
        } else
            break;
    }

    return currentcond;
}

} }   // namespace skylark::ml

#endif // SKYLARK_LOCAL_COMPUTATIONS_HPP
