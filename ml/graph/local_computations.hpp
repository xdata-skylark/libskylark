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
    const std::unordered_map<typename GraphType::vertex_type, T>& s,
    std::unordered_map<typename GraphType::vertex_type, El::Matrix<T> *>& y,
    El::Matrix<T> &x, double alpha, double gamma, double epsilon, int NX = 4) {

    typedef typename GraphType::vertex_type vertex_type;

    if (!El::Initialized())
        SKYLARK_THROW_EXCEPTION (
            base::skylark_exception()
               << base::error_msg("Elemental was not initialized") );

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
    const El::Int N = minN % NX == 0 ? minN : (minN / NX + 1) * NX;
    const El::Int NR = N / NX;

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

    El::Matrix<T> x1;
    nla::ChebyshevPoints(N, x1, 0, gamma);
    x.Resize(NX, 1);
    for(int i = 0; i < NX; i++)
        x.Set(i, 0, x1.Get(i * NR, 0));

    // Constants for convergence.
    double LC = 1 + (2 / pi) * log(N - 1);
    double C = (alpha < 1) ?
        (1-alpha) * epsilon / ((1 - exp((alpha - 1) * gamma)) * LC) :
        epsilon / (gamma * LC);

    // From now on, do not use Elemental to avoid overheads.
    const T *D = D_->LockedBuffer();
    const T *u = D_->LockedBuffer() + N - 1;

    typedef std::pair<bool, T*> rypair_t;
    std::unordered_map<vertex_type, rypair_t> rymap;
    std::queue<vertex_type> violating;

    // Initialize non-zero functions, and their residual, which is not
    // fully computed yet (but we know that needs to be inserted into the queue).
    for(auto it = s.begin(); it != s.end(); it++) {
        const vertex_type &node = it->first;
        const T &v = it->second;

        T *ry = new T[N + NX];
        std::fill(ry, ry + N, -alpha * v);
        std::fill(ry + N, ry + N + NX, v);
        rymap[node] = rypair_t(true, ry);
        violating.push(node);
    }

    // Initialize to just zero for all nodes adjanct to seeds, that
    // are not seeds themselves. Residual is not fully computed yet.
    for(auto it = s.begin(); it != s.end(); it++) {
        const vertex_type &node = it->first;

        for(auto it = G.adjanct_begin(node); it != G.adjanct_end(node); it++) {
            const vertex_type &onode = *it;
            if (rymap.count(onode) == 0) {
                T *ry = new T[N + NX];
                std::fill(ry, ry + N + NX, 0);
                rymap[onode] = rypair_t(false, ry);
            }
        }
    }

    // Update the residual based on seeds
    for(auto it = s.begin(); it != s.end(); it++) {
        const vertex_type &node = it->first;

        T *ry = rymap[node].second;

        size_t deg = G.degree(node);
        T v = alpha * ry[N] / deg;
        for(auto it = G.adjanct_begin(node); it != G.adjanct_end(node); it++) {
            const vertex_type &onode = *it;
            size_t odeg = G.degree(onode);

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
        vertex_type node = violating.front();
        violating.pop();

        // Solve locally, and update rymap[node].
        rypair_t& rpair = rymap[node];
        T *ry = rpair.second;

        // Compute correction to y, and the new residual.
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
        size_t deg = G.degree(node);
        for(auto it = G.adjanct_begin(node); it != G.adjanct_end(node); it++) {
            const vertex_type &onode = *it;
            size_t odeg = G.degree(onode);

            // Add it to rymap, if not already there.
            if (rymap.count(onode) == 0) {
                T *rynew = new T[N + NX];
                std::fill(rynew, rynew + N + NX, 0);
                rymap[onode] = rypair_t(false, rynew);
            }

            rypair_t& ryopair = rymap[onode];
            bool inq = false;
            T *ryo = ryopair.second;
            T c = alpha / deg;
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

    // Yank values to y, freeing other parts of ry in the process.
    y.clear();
    for(auto it = rymap.begin(); it != rymap.end(); it++) {
        if (it->second.second[N] != 0) {
            El::Matrix<T> *yv = new El::Matrix<T>(NX, 1);
            for(int i = 0; i < NX; i++)
                yv->Set(i, 0, it->second.second[N + i]);
            y[it->first] = yv;
        }
        delete it->second.second;
    }
}

template<typename GraphType>
double FindLocalCluster(const GraphType& G,
    const std::unordered_set<typename GraphType::vertex_type>& seeds,
    std::unordered_set<typename GraphType::vertex_type>& cluster,
    double alpha = 0.85, double gamma = 5.0, double epsilon = 0.001, int NX = 4,
    bool recursive = true) {

    typedef typename GraphType::vertex_type vertex_type;
    double currentcond = -1;
    cluster = seeds;
    bool improve;
    El::Matrix<double> x;

    do {
        // Create seed set.
        std::unordered_map<vertex_type, double> s;
        for(auto it = seeds.begin(); it != seeds.end(); it++)
            s[*it] = 1.0 / seeds.size();

        // Run the diffusion
        std::unordered_map<vertex_type, El::Matrix<double>*> y;
        LocalGraphDiffusion(G, s, y, x, alpha, gamma, epsilon, NX);

        // Go over the y output at the different time samples,
        // find the best prefix and if better conductance, store it.
        improve = false;
        for (int t = 0; t < NX; t++) {
            // Sort (descending) the non-zero components based on their normalized
            // y values (normalized by degree).
            std::vector<std::pair<double, vertex_type> > vals(y.size());
            int i = 0;
            for(auto it = y.begin(); it != y.end(); it++) {
                vertex_type node = it->first;
                double val = - it->second->Get(t, 0) / G.degree(node);
                vals[i] = std::make_pair(val, node);
                i++;
            }
            std::sort(vals.begin(), vals.end());

            // Find the best prefix
            int volS = 0, cutS = 0;
            double bestcond = 1.0;
            int bestprefix = 0;
            int Gvol = G.num_edges();
            std::unordered_set<vertex_type> currentset;
            for (int i = 0; i < vals.size(); i++) {
                vertex_type node = vals[i].second;
                size_t deg = G.degree(node);
                volS += deg;
                for(auto it = G.adjanct_begin(node);
                    it != G.adjanct_end(node); it++) {
                    const vertex_type &onode = *it;
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
                    cluster.insert(vals[i].second);
                currentcond = bestcond;
            }
        }

        // Clear y
        for(auto it = y.begin(); it != y.end(); it++)
            delete it->second;
    } while (recursive && improve);

    return currentcond;
}

} }   // namespace skylark::ml

#endif // SKYLARK_LOCAL_COMPUTATIONS_HPP
