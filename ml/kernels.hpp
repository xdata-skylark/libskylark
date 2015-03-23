#ifndef SKYLARK_KERNELS_HPP
#define SKYLARK_KERNELS_HPP

#include "../sketch/sketch.hpp"
#include "feature_transform_tags.hpp"

namespace skylark { namespace ml { 

struct gaussian_t {

    gaussian_t(int N, double sigma) : _N(N), _sigma(sigma) {

    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::GaussianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        fast_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::FastGaussianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT, 
             template <typename> class QMCSequenceType>
    sketch::sketch_transform_t<IT, OT> *create_qrft(int S,
        const QMCSequenceType<double>& sequence, int skip,
        base::context_t& context) const {
        return
            new sketch::GaussianQRFT_t<IT, OT, QMCSequenceType>(_N,
                S, _sigma, sequence, skip, context);
    }

    int get_dim() const {
        return _N;
    }

    int qrft_sequence_dim() const {
        return sketch::GaussianQRFT_data_t<base::qmc_sequence_container_t>::
            qmc_sequence_dim(_N);
    }

    template<typename XT, typename YT, typename KT>
    friend void Gram(base::direction_t dirX, base::direction_t dirY,
        const gaussian_t& k, const XT &X, const YT &Y, KT &K);

    template<typename XT, typename KT>
    friend void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
        const gaussian_t& k, const XT &X, KT &K);

    template<typename XT, typename YT, typename KT>
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) {

        int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        K.Resize(m, n);
        base::Euclidean(dirX, dirY, 1.0, X, Y, 0.0, K);
        typedef typename utility::typer_t<KT>::value_type value_type;
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::exp(-x / (2 * _sigma * _sigma));
            }));
    }

private:
    const int _N;
    const double _sigma;
};

template<typename XT, typename YT, typename KT>
void Gram(base::direction_t dirX, base::direction_t dirY,
    const gaussian_t& k, const XT &X, const YT &Y, KT &K) {

    int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
    int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

    K.Resize(m, n);
    base::Euclidean(dirX, dirY, 1.0, X, Y, 0.0, K);
    typedef typename utility::typer_t<KT>::value_type value_type;
    El::EntrywiseMap(K, std::function<value_type(value_type)> (
          [k] (value_type x) {
              return std::exp(-x / (2 * k._sigma * k._sigma));
          }));
}

template<typename XT, typename KT>
void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const gaussian_t& k, const XT &X, KT &K) {

    int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);

    K.Resize(n, n);
    base::SymmetricEuclidean(uplo, dir, 1.0, X, 0.0, K);
    typedef typename utility::typer_t<KT>::value_type value_type;
    El::EntrywiseMap(K, std::function<value_type(value_type)> (
          [k] (value_type x) {
              return std::exp(-x / (2 * k._sigma * k._sigma));
          }));
}

struct polynomial_t {

    polynomial_t(int N, int q = 2, double c = 1.0, double gamma = 1.0)
        : _N(N), _q(q), _c(c), _gamma(gamma) {

    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::PPT_t<IT, OT>(_N, S, _q, _c, _gamma, context);
    }

    int get_dim() const {
        return _N;
    }

    // TODO method for gram matrix ?


private:
    const int _N;
    const int _q;
    const double _c;
    const double _gamma;
};

struct laplacian_t {

    laplacian_t(int N, double sigma) : _N(N), _sigma(sigma) {

    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::LaplacianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT,
             template<typename> class QMCSequenceType>
    sketch::sketch_transform_t<IT, OT> *create_qrft(int S,
        const QMCSequenceType<double>& sequence, int skip,
        base::context_t& context) const {
        return
            new sketch::LaplacianQRFT_t<IT, OT, QMCSequenceType>(_N,
                S, _sigma, sequence, skip, context);
    }

    int get_dim() const {
        return _N;
    }

    int qrft_sequence_dim() const {
        return sketch::LaplacianQRFT_data_t<base::qmc_sequence_container_t>::
            qmc_sequence_dim(_N);
    }

    // TODO method for gram matrix ?


private:
    const int _N;
    const double _sigma;
};

struct expsemigroup_t {

    expsemigroup_t(int N, double beta) : _N(N), _beta(beta) {

    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::ExpSemigroupRLT_t<IT, OT>(_N, S, _beta, context);
    }

    template<typename IT, typename OT,
             template<typename> class QMCSequenceType>
    sketch::sketch_transform_t<IT, OT> *create_qrft(int S,
        const QMCSequenceType<double>& sequence, int skip,
        base::context_t& context) const {
        return
            new sketch::ExpSemigroupQRLT_t<IT, OT, QMCSequenceType>(_N,
                S, _beta, sequence, skip, context);
    }

    int get_dim() const {
        return _N;
    }

    int qrft_sequence_dim() const {
        return sketch::ExpSemigroupQRLT_data_t<base::qmc_sequence_container_t>::
            qmc_sequence_dim(_N);
    }

    // TODO method for gram matrix ?


private:
    const int _N;
    const double _beta;
};

struct matern_t {

    matern_t(int N, double nu, double l) : _N(N), _nu(nu), _l(l) {

    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::MaternRFT_t<IT, OT>(_N, S, _nu, _l, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        fast_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::FastMaternRFT_t<IT, OT>(_N, S, _nu, _l, context);
    }


    int get_dim() const {
        return _N;
    }

    // TODO method for gram matrix ?


private:
    const int _N;
    const double _nu;
    const double _l;
};

} } 

#endif // SKYLARK_KERNELS_HPP
