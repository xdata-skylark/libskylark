#ifndef SKYLARK_SBM_HPP
#define SKYLARK_SBM_HPP

#include <skylark.hpp>

namespace skysketch = skylark::sketch;
namespace skyml = skylark::ml;
namespace skynla = skylark::nla;
namespace skyutil = skylark::utility;

namespace {
template <typename T>
T power2(T arg) {
    return arg*arg;
}

template <typename T>
T sqrtfn(const T arg) {
    return pow(arg, 0.5);
}

// TODO: Untested
template <typename T>
void col_division(El::Matrix<T> numer, const El::Matrix<T> denom) {
    assert(numer.Height() == denom.Width());

    // TODO: Bad for cpu cache for both numer & denom
    for (El::Unsigned col = 0; col < numer.Width(); col++) {
        for (El::Unsigned row = 0; row < numer.Height(); row++) {
            numer.Set(row, col, (numer.Get(row, col) /
                    denom.Get(0, row)));
        }
    }
}

// TODO: Untested
template <typename T>
void get_matrix_row(El::DistMatrix<T>& mat, const El::Unsigned rid,
        std::vector<T>& ret) {
    for (El::Unsigned col = 0; col < mat.Width(); col++) {
        ret.push_back(mat.Get(rid, col));
    }
}

// TODO: Untested
template <typename T>
skyml::kmeans_t<T> get_clusters(El::DistMatrix<T>& _X_hat,
        const El::Unsigned k, const El::Unsigned nelbows=3) {
    El::DistMatrix<T> X_hat = _X_hat; // Make a copy
    std::function<T(T)> pow2func = power2<T>;
    El::EntrywiseMap(_X_hat, pow2func); // Square each entry

    // colSums
    El::AllReduce(_X_hat, El::mpi::COMM_WORLD, El::mpi::SUM);
    // TODO: sqrt
    std::function<T(T)> sqrtfunc = sqrtfn<T>;
    El::EntrywiseMap(_X_hat, sqrtfunc); // FIXME: Wasteful

    std::vector<T> eigvals;
    get_matrix_row<T>(_X_hat, 0, eigvals); // All rows should be the same
    El::Unsigned d_hat = skyutil::get_elbows<T>(eigvals, nelbows)[0];

    // Create scaled_X_hat
    // Numerator
    El::DistMatrix<T> numerator_scaled_X_hat(X_hat.Height(), d_hat);
    El::GetSubmatrix(X_hat, El::IR(0, X_hat.Height()),
                El::IR(0, d_hat), numerator_scaled_X_hat);

    El::DistMatrix<T> denom_scaled_X_hat;
    El::Transpose(numerator_scaled_X_hat, denom_scaled_X_hat); // t(X_hat_slice)
    El::EntrywiseMap(denom_scaled_X_hat, pow2func); // Square each entry
    // rowSums
    El::AllReduce(denom_scaled_X_hat, El::mpi::COMM_WORLD, El::mpi::SUM);
    El::EntrywiseMap(denom_scaled_X_hat, sqrtfunc);

    // Now we must divide numerator by denominator so get data to root
    // TODO: Serial
    El::DistMatrix<T, El::CIRC, El::CIRC> scaled_X_hat_local =
        numerator_scaled_X_hat;
    // Get all data to root
    El::DistMatrix<T, El::CIRC, El::CIRC> denom_scaled_X_hat_local =
        denom_scaled_X_hat;
    if (scaled_X_hat_local.DistRank() == 0) {
        // Divide edits scaled_X_hat_local
        col_division(scaled_X_hat_local.Matrix(),
                denom_scaled_X_hat.LockedMatrix());
    }
    // TODO: End Serial
    El::mpi::Barrier();

    // Run kmeans on data
    El::DistMatrix<T, El::VC, El::STAR> scaled_X_hat =
        scaled_X_hat_local;
    El::Matrix<T> centroids(k, d_hat);

    return skyml::run_kmeans<T>(scaled_X_hat, centroids, k, 1E-6,
            "random", 1, 50, scaled_X_hat.DistRank());
}
} // End Annoymous namespace

namespace skylark { namespace ml {

/** Implements https://arxiv.org/pdf/1503.02115v3.pdf, but with sketching and
 *   and Randomized power iteration SVD instead of adjacency spectral embedding
**/
template <typename T>
const skyml::kmeans_t<T> sketched_sbm_clusters(const El::DistMatrix<T>& A,
        const El::Unsigned d, const El::Unsigned k,
        const unsigned powerits=2) {
    // Initializing context
    skylark::base::context_t context(0);

    // Sketch A
    El::DistMatrix<T> X_hat(A.Height(), d);

    skysketch::JLT_t<El::DistMatrix<T>, El::DistMatrix<T> >
        sketch_transform(A.Width(), d, context);
    sketch_transform.apply(A, X_hat, skysketch::rowwise_tag());

    // QR factorization
    El::DistMatrix<T> d_mat;
    El::DistMatrix<T> t_mat;
    // NOTE: X_hat is gone and is now Q
    El::QR(X_hat, t_mat, d_mat); // TODO: Test X_hat now has Q (not Qstar)

    // SVD step
    El::DistMatrix<T> B;
    El::Gemm(El::TRANSPOSE, El::NORMAL, T(1), X_hat, A, T(0), B); // Qstar * A

    skynla::approximate_svd_params_t params;
    params.num_iterations = powerits;

    // Make SVD output structs
    El::DistMatrix<T> U, S, V;
    skynla::ApproximateSVD(B, U, S, V, d, context, params);

    // TODO
    /*
       singvecs = X_hat * U
       singvals = sqrt(ab(S))
       nsingvals = S.width()
       retXhat \in A.height(), d

       for j in range(d):
           X_hat[:, j] = singvecs[:, nsingvals-j] * singvals[nsingvals-j]
    */

    return get_clusters<T>(X_hat, k);
}
} } // namespace skylark::ml
#endif // SKYLARK_SBM_HPP
