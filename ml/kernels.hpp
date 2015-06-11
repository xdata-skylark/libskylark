#ifndef SKYLARK_KERNELS_HPP
#define SKYLARK_KERNELS_HPP

#include "../sketch/sketch.hpp"
#include "feature_transform_tags.hpp"

namespace skylark { namespace ml { 

struct linear_t {

    linear_t(int N) : _N(N) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "linear");
        pt.put("N", _N);

        return pt;
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::JLT_t<IT, OT>(_N, S, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        fast_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::FJLT_t<IT, OT>(_N, S, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(int S,
        sparse_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::CWT_t<IT, OT>(_N, S, context);
    }


    int get_dim() const {
        return _N;
    }

private:

    int _N;
};

struct gaussian_t {

    gaussian_t(El::Int N, double sigma) : _N(N), _sigma(sigma) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "gaussian");
        pt.put("sigma", _sigma);
        pt.put("N", _N);

        return pt;
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::GaussianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        fast_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::FastGaussianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT, 
             template <typename> class QMCSequenceType>
    sketch::sketch_transform_t<IT, OT> *create_qrft(El::Int S,
        const QMCSequenceType<double>& sequence, int skip,
        base::context_t& context) const {
        return
            new sketch::GaussianQRFT_t<IT, OT, QMCSequenceType>(_N,
                S, _sigma, sequence, skip, context);
    }

    El::Int get_dim() const {
        return _N;
    }

    El::Int qrft_sequence_dim() const {
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

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        K.Resize(m, n);
        base::EuclideanDistanceMatrix(dirX, dirY, 1.0, X, Y, 0.0, K);
        typedef typename utility::typer_t<KT>::value_type value_type;
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::exp(-x / (2 * _sigma * _sigma));
            }));
    }

private:
    const El::Int _N;
    const double _sigma;
};

template<typename XT, typename YT, typename KT>
void Gram(base::direction_t dirX, base::direction_t dirY,
    const gaussian_t& k, const XT &X, const YT &Y, KT &K) {

    typedef typename utility::typer_t<KT>::value_type value_type;

    El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
    El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

    K.Resize(m, n);
    base::EuclideanDistanceMatrix(dirX, dirY, value_type(1.0), X, Y,
        value_type(0.0), K);
    El::EntrywiseMap(K, std::function<value_type(value_type)> (
          [k] (value_type x) {
              return std::exp(-x / (2 * k._sigma * k._sigma));
          }));
}

template<typename XT, typename KT>
void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const gaussian_t& k, const XT &X, KT &K) {

    typedef typename utility::typer_t<KT>::value_type value_type;

    El::Int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);

    K.Resize(n, n);
    base::SymmetricEuclideanDistanceMatrix(uplo, dir, value_type(1.0), X,
        value_type(0.0), K);
    base::SymmetricEntrywiseMap(uplo, K, std::function<value_type(value_type)> (
          [k] (value_type x) {
              return std::exp(-x / (2 * k._sigma * k._sigma));
          }));
}

struct polynomial_t {

    polynomial_t(El::Int N, int q = 2, double c = 1.0, double gamma = 1.0)
        : _N(N), _q(q), _c(c), _gamma(gamma) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "polynomial");
        pt.put("q", _q);
        pt.put("c", _c);
        pt.put("gamma", _gamma);
        pt.put("N", _N);

        return pt;
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::PPT_t<IT, OT>(_N, S, _q, _c, _gamma, context);
    }

    El::Int get_dim() const {
        return _N;
    }

    // TODO method for gram matrix ?


private:
    const El::Int _N;
    const int _q;
    const double _c;
    const double _gamma;
};

struct laplacian_t {

    laplacian_t(El::Int N, double sigma) : _N(N), _sigma(sigma) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "laplacian");
        pt.put("sigma", _sigma);
        pt.put("N", _N);

        return pt;
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::LaplacianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT,
             template<typename> class QMCSequenceType>
    sketch::sketch_transform_t<IT, OT> *create_qrft(El::Int S,
        const QMCSequenceType<double>& sequence, El::Int skip,
        base::context_t& context) const {
        return
            new sketch::LaplacianQRFT_t<IT, OT, QMCSequenceType>(_N,
                S, _sigma, sequence, skip, context);
    }

    El::Int get_dim() const {
        return _N;
    }

    El::Int qrft_sequence_dim() const {
        return sketch::LaplacianQRFT_data_t<base::qmc_sequence_container_t>::
            qmc_sequence_dim(_N);
    }

    template<typename XT, typename YT, typename KT>
    friend void Gram(base::direction_t dirX, base::direction_t dirY,
        const laplacian_t& k, const XT &X, const YT &Y, KT &K);

    template<typename XT, typename KT>
    friend void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
        const laplacian_t& k, const XT &X, KT &K);

    template<typename XT, typename YT, typename KT>
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) {

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        K.Resize(m, n);
        base::L1DistanceMatrix(dirX, dirY, 1.0, X, Y, 0.0, K);
        typedef typename utility::typer_t<KT>::value_type value_type;
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::exp(-x / _sigma);
            }));
    }

private:
    const El::Int _N;
    const double _sigma;
};

template<typename XT, typename YT, typename KT>
void Gram(base::direction_t dirX, base::direction_t dirY,
    const laplacian_t& k, const XT &X, const YT &Y, KT &K) {

    typedef typename utility::typer_t<KT>::value_type value_type;

    El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
    El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

    K.Resize(m, n);
    base::L1DistanceMatrix(dirX, dirY, value_type(1.0), X, Y,
        value_type(0.0), K);
    El::EntrywiseMap(K, std::function<value_type(value_type)> (
          [k] (value_type x) {
              return std::exp(-x / k._sigma);
          }));
}

template<typename XT, typename KT>
void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const laplacian_t& k, const XT &X, KT &K) {

    typedef typename utility::typer_t<KT>::value_type value_type;

    El::Int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);

    K.Resize(n, n);
    base::SymmetricL1DistanceMatrix(uplo, dir, value_type(1.0), X,
        value_type(0.0), K);
    base::SymmetricEntrywiseMap(uplo, K, std::function<value_type(value_type)> (
          [k] (value_type x) {
              return std::exp(-x / k._sigma);
          }));
}

struct expsemigroup_t {

    expsemigroup_t(El::Int N, double beta) : _N(N), _beta(beta) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "expsemigroup");
        pt.put("beta", _beta);
        pt.put("N", _N);

        return pt;
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::ExpSemigroupRLT_t<IT, OT>(_N, S, _beta, context);
    }

    template<typename IT, typename OT,
             template<typename> class QMCSequenceType>
    sketch::sketch_transform_t<IT, OT> *create_qrft(El::Int S,
        const QMCSequenceType<double>& sequence, El::Int skip,
        base::context_t& context) const {
        return
            new sketch::ExpSemigroupQRLT_t<IT, OT, QMCSequenceType>(_N,
                S, _beta, sequence, skip, context);
    }

    El::Int get_dim() const {
        return _N;
    }

    El::Int qrft_sequence_dim() const {
        return sketch::ExpSemigroupQRLT_data_t<base::qmc_sequence_container_t>::
            qmc_sequence_dim(_N);
    }

    // TODO method for gram matrix ?


private:
    const El::Int _N;
    const double _beta;
};

struct matern_t {

    matern_t(El::Int N, double nu, double l) : _N(N), _nu(nu), _l(l) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "matern");
        pt.put("nu", _nu);
        pt.put("l", _l);
        pt.put("N", _N);

        return pt;
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::MaternRFT_t<IT, OT>(_N, S, _nu, _l, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        fast_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::FastMaternRFT_t<IT, OT>(_N, S, _nu, _l, context);
    }


    El::Int get_dim() const {
        return _N;
    }

    // TODO method for gram matrix ?


private:
    const El::Int _N;
    const double _nu;
    const double _l;
};

} } 

#endif // SKYLARK_KERNELS_HPP
