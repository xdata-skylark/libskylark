#ifndef SKYLARK_KERNELS_HPP
#define SKYLARK_KERNELS_HPP

#include "../sketch/sketch.hpp"
#include "feature_transform_tags.hpp"

namespace skylark { namespace ml {

/**
 * Base class for all kernels.
 */
struct kernel_t {

    virtual ~kernel_t() {

    }

    virtual El::Int get_dim() const = 0;

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const = 0;

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const = 0;

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const = 0;

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const = 0;

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<double> &X, El::Matrix<double> &K) const = 0;

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const = 0;

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const = 0;

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const = 0;

    virtual
    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const = 0;

    virtual
    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
        fast_feature_transform_tag, base::context_t& context) const = 0;

    template<typename IT, typename OT, typename TT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        TT tag, base::context_t& context) const {

        sketch::generic_sketch_transform_ptr_t p(create_rft(S, tag, context));
        return new sketch::sketch_transform_container_t<IT, OT>(p);
    }

    virtual boost::property_tree::ptree to_ptree() const = 0;
};

template<typename Kernel, typename XT, typename YT, typename KT>
void Gram(base::direction_t dirX, base::direction_t dirY,
    const Kernel& k, const XT &X, const YT &Y, KT &K) {

    k.gram(dirX, dirY, X, Y, K);
}

void Gram(base::direction_t dirX, base::direction_t dirY,
    const kernel_t &k, const boost::any &X, const boost::any &Y,
    const boost::any &K) {

#define SKYLARK_GRAM_ANY_APPLY_DISPATCH(XT, YT, KT)                     \
    if (K.type() == typeid(KT*)) {                                      \
        if (X.type() == typeid(XT*)) {                                  \
            if (Y.type() == typeid(YT*)) {                              \
                Gram(dirX, dirY, k, *boost::any_cast<XT*>(X),           \
                    *boost::any_cast<YT*>(Y),                           \
                    *boost::any_cast<KT*>(K));                          \
                return;                                                 \
            }                                                           \
                                                                        \
            if (Y.type() == typeid(const YT*)) {                        \
                Gram(dirX, dirY, k, *boost::any_cast<XT*>(X),           \
                    *boost::any_cast<const YT*>(Y),                     \
                    *boost::any_cast<KT*>(K));                          \
                return;                                                 \
            }                                                           \
        }                                                               \
        if (X.type() == typeid(const XT*)) {                            \
            if (Y.type() == typeid(YT*)) {                              \
                Gram(dirX, dirY, k, *boost::any_cast<const XT*>(X),     \
                    *boost::any_cast<YT*>(Y),                           \
                    *boost::any_cast<KT*>(K));                          \
                return;                                                 \
            }                                                           \
                                                                        \
            if (Y.type() == typeid(const YT*)) {                        \
                Gram(dirX, dirY, k, *boost::any_cast<const XT*>(X),     \
                    *boost::any_cast<const YT*>(Y),                     \
                    *boost::any_cast<KT*>(K));                          \
                return;                                                 \
            }                                                           \
        }                                                               \
    }

#if !(defined SKYLARK_NO_ANY)

    SKYLARK_GRAM_ANY_APPLY_DISPATCH(mdtypes::matrix_t,
        mdtypes::matrix_t, mdtypes::matrix_t);
    SKYLARK_GRAM_ANY_APPLY_DISPATCH(mdtypes::el_matrix_t,
        mdtypes::el_matrix_t, mdtypes::el_matrix_t);

    SKYLARK_GRAM_ANY_APPLY_DISPATCH(mftypes::matrix_t,
        mftypes::matrix_t, mftypes::matrix_t);
    SKYLARK_GRAM_ANY_APPLY_DISPATCH(mftypes::el_matrix_t,
        mftypes::el_matrix_t, mftypes::el_matrix_t);
#endif
    
    SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "This combination has not yet been implemented for Gram"));
#undef SKYLARK_GRAM_ANY_APPLY_DISPATCH
}

template<typename Kernel, typename XT, typename KT>
void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const Kernel& k, const XT &X, KT &K) {

    k.symmetric_gram(uplo, dir, X, K);
}

void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const kernel_t& k, const boost::any &X, const boost::any &K) {

    // TODO
    SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "SymmetricGram has not yet been implemented for boost::any params"));
}

/**
 * Linear kernel: simple linear product.
 */
struct linear_t : public kernel_t {

    linear_t(El::Int N) : _N(N) {

    }

    linear_t(const boost::property_tree::ptree &pt) :
        _N(pt.get<El::Int>("N")) {

    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "kernel");
        pt.put("skylark_version", VERSION);

        pt.put("kernel_type", "linear");
        pt.put("N", _N);

        return pt;
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    regular_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    fast_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::JLT_t<IT, OT>(_N, S, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        fast_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::FJLT_t<IT, OT>(_N, S, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        sparse_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::CWT_t<IT, OT>(_N, S, context);
    }

    El::Int get_dim() const {
        return _N;
    }

    template<typename XT, typename YT, typename KT>
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        El::Orientation xo = dirX == base::COLUMNS ? El::ADJOINT : El::NORMAL;
        El::Orientation yo = dirY == base::COLUMNS ? El::NORMAL : El::ADJOINT;

        K.Resize(m, n);
        El::Gemm(xo, yo, value_type(1.0), X, Y, value_type(0.0), K);
    }

    template<typename XT, typename KT>
    void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const XT &X, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Orientation o = dir == base::COLUMNS ? El::ADJOINT : El::NORMAL;

        K.Resize(n, n);
        El::Herk(uplo, o, value_type(1.0), X, K);
    }


     /* Instantion of virtual functions in base */

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir, 
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);

    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

private:

    El::Int _N;
};

/**
 * Gaussian kernel.
 */
struct gaussian_t : public kernel_t {

    gaussian_t(El::Int N, double sigma) : _N(N), _sigma(sigma) {

    }

    gaussian_t(const boost::property_tree::ptree &pt) :
        _N(pt.get<El::Int>("N")), _sigma(pt.get<double>("sigma")) {

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

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    regular_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    fast_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
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
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        K.Resize(m, n);
        base::EuclideanDistanceMatrix(dirX, dirY, value_type(1.0), X, Y,
            value_type(0.0), K);
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::exp(-x / (2 * _sigma * _sigma));
            }));
    }

    template<typename XT, typename KT>
    void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const XT &X, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);

        K.Resize(n, n);
        base::SymmetricEuclideanDistanceMatrix(uplo, dir, value_type(1.0), X,
            value_type(0.0), K);
        base::SymmetricEntrywiseMap(uplo, K, std::function<value_type(value_type)> (
              [this] (value_type x) {
                  return std::exp(-x / (2 * _sigma * _sigma));
              }));
    }

    /* Instantion of virtual functions in base */

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir, 
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);

    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

private:
    const El::Int _N;
    const double _sigma;
};

struct polynomial_t : public kernel_t {

    polynomial_t(El::Int N, int q = 2, double c = 1.0, double gamma = 1.0)
        : _N(N), _q(q), _c(c), _gamma(gamma) {

    }

    polynomial_t(const boost::property_tree::ptree &pt) :
        _N(pt.get<El::Int>("N")), _q(pt.get<int>("q")), _c(pt.get<double>("c")),
        _gamma(pt.get<double>("gamma")) {

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

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    regular_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    fast_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        regular_feature_transform_tag, base::context_t& context) const {
        return
            new sketch::PPT_t<IT, OT>(_N, S, _q, _c, _gamma, context);
    }

    template<typename IT, typename OT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        fast_feature_transform_tag, base::context_t& context) const {
        // PPT is also "fast"
        return
            new sketch::PPT_t<IT, OT>(_N, S, _q, _c, _gamma, context);
    }

    El::Int get_dim() const {
        return _N;
    }

    template<typename XT, typename YT, typename KT>
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        El::Orientation xo = dirX == base::COLUMNS ? El::ADJOINT : El::NORMAL;
        El::Orientation yo = dirY == base::COLUMNS ? El::NORMAL : El::ADJOINT;

        K.Resize(m, n);
        // TODO should be base::Gemm, not El::Gemm
        El::Gemm(xo, yo, value_type(1.0), X, Y, value_type(0.0), K);
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::pow(_gamma * x + _c, _q);
            }));
    }

    template<typename XT, typename KT>
    void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const XT &X, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Orientation o = dir == base::COLUMNS ? El::ADJOINT : El::NORMAL;

        K.Resize(n, n);
        // TODO should be base::Herk, not El::Herk
        El::Herk(uplo, o, value_type(1.0), X, K);
        // TODO maybe need to use Gemm version.
        base::SymmetricEntrywiseMap(uplo, K, std::function<value_type(value_type)> (
              [this] (value_type x) {
                  return std::pow(_gamma * x + _c, _q);
              }));
    }


    /* Instantion of virtual functions in base */

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir, 
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);

    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

private:
    const El::Int _N;
    const int _q;
    const double _c;
    const double _gamma;
};

/**
 * Laplacian kernel
 */
struct laplacian_t : public kernel_t {

    laplacian_t(El::Int N, double sigma) : _N(N), _sigma(sigma) {

    }

    laplacian_t(const boost::property_tree::ptree &pt) :
        _N(pt.get<El::Int>("N")), _sigma(pt.get<double>("sigma")) {

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

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    regular_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    fast_feature_transform_tag tag, base::context_t& context) const {

        // TODO
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "fast_feature_transform has not yet been implemented for laplacian kernel"));
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
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        K.Resize(m, n);
        base::L1DistanceMatrix(dirX, dirY, value_type(1.0), X, Y,
            value_type(0.0), K);
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::exp(-x / _sigma);
            }));
    }

    template<typename XT, typename KT>
    void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const XT &X, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int n = dir == base::COLUMNS ? base::Width(X) : base::Height(X);

        K.Resize(n, n);
        base::SymmetricL1DistanceMatrix(uplo, dir, value_type(1.0), X,
            value_type(0.0), K);
        base::SymmetricEntrywiseMap(uplo, K, std::function<value_type(value_type)> (
              [this] (value_type x) {
                  return std::exp(-x / _sigma);
              }));
    }

    /* Instantion of virtual functions in base */

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir, 
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);

    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

private:
    const El::Int _N;
    const double _sigma;
};


struct expsemigroup_t : public kernel_t {

    expsemigroup_t(El::Int N, double beta) : _N(N), _beta(beta) {

    }

    expsemigroup_t(const boost::property_tree::ptree &pt) :
        _N(pt.get<El::Int>("N")), _beta(pt.get<double>("beta")) {

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

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    regular_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    fast_feature_transform_tag tag, base::context_t& context) const {
        // TODO
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "fast_feature_transform has not yet been implemented for expsemigroup kernel"));
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

    template<typename XT, typename YT, typename KT>
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) const {

        typedef typename utility::typer_t<KT>::value_type value_type;

        El::Int m = dirX == base::COLUMNS ? base::Width(X) : base::Height(X);
        El::Int n = dirY == base::COLUMNS ? base::Width(Y) : base::Height(Y);

        K.Resize(m, n);
        base::ExpsemigroupDistanceMatrix(dirX, dirY, value_type(1.0), X, Y,
            value_type(0.0), K);
        El::EntrywiseMap(K, std::function<value_type(value_type)> (
            [this] (value_type x) {
                return std::exp(-_beta * x);
            }));
    }

    template<typename XT, typename KT>
    void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const XT &X, KT &K) const {

        // TODO: Implement the kernel for symmetric matrices
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "symmetric_gram has not yet been implemented for expsemigroup kernel"));
    }

    /* Instantion of virtual functions in base */

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir, 
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);

    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }


private:
    const El::Int _N;
    const double _beta;
};

struct matern_t : kernel_t {

    matern_t(El::Int N, double nu, double l) : _N(N), _nu(nu), _l(l) {

    }

    matern_t(const boost::property_tree::ptree &pt) :
        _N(pt.get<El::Int>("N")), _nu(pt.get<double>("nu")),
        _l(pt.get<double>("l")) {

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

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    regular_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);
    }

    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
    fast_feature_transform_tag tag, base::context_t& context) const {

        return create_rft<boost::any, boost::any>(S, tag, context);

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

    template<typename XT, typename YT, typename KT>
    void gram(base::direction_t dirX, base::direction_t dirY,
        const XT &X, const YT &Y, KT &K) const {

        // TODO: Implement the kernel
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "gram has not yet been implemented for matern kernel"));
    }

    template<typename XT, typename KT>
    void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const XT &X, KT &K) const {

        // TODO: Implement the kernel for symmetric matrices
        SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "symmetric_gram has not yet been implemented for matern kernel"));
    }

    /* Instantion of virtual functions in base */

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        gram<matrix_type, matrix_type, matrix_type>(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir, 
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        typedef El::Matrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);

    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        typedef El::Matrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        typedef El::ElementalMatrix<double> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        typedef El::ElementalMatrix<float> matrix_type;
        symmetric_gram<matrix_type, matrix_type>(uplo, dir, X, K);
    }


private:
    const El::Int _N;
    const double _nu;
    const double _l;
};

/**
 * Container of a kernel that supports the interface.
 * Useful for keeping copies of the kernel inside objects.
 */
struct kernel_container_t : public kernel_t {

    kernel_container_t(const std::shared_ptr<kernel_t> k) :
        _k(k) {
    }

    kernel_container_t(const boost::property_tree::ptree &pt) {
        std::string type = pt.get<std::string>("kernel_type");
        
        _type = type;

        if (type == "linear")
            _k.reset(new linear_t(pt));
        else if (type == "gaussian")
            _k.reset(new gaussian_t(pt));
        else if (type == "laplacian")
            _k.reset(new laplacian_t(pt));
        else if (type == "polynomial")
            _k.reset(new polynomial_t(pt));
        else if (type == "expsemigroup")
            _k.reset(new expsemigroup_t(pt));
        else if (type == "matern")
            _k.reset(new matern_t(pt));
    }

    kernel_container_t() {

    }

    virtual ~kernel_container_t() {

    }

    virtual El::Int get_dim() const { return _k->get_dim(); }

    virtual std::string get_type() const {
        return _type;
    }

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<double> &X, const El::Matrix<double> &Y,
        El::Matrix<double> &K) const {

        _k->gram(dirX, dirY, X, Y, K);
    }

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::Matrix<float> &X, const El::Matrix<float> &Y,
        El::Matrix<float> &K) const {

        _k->gram(dirX, dirY, X, Y, K);
    }

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<double> &X,
        const El::ElementalMatrix<double> &Y,
        El::ElementalMatrix<double> &K) const {

        _k->gram(dirX, dirY, X, Y, K);
    }

    virtual void gram(base::direction_t dirX, base::direction_t dirY,
        const El::ElementalMatrix<float> &X,
        const El::ElementalMatrix<float> &Y,
        El::ElementalMatrix<float> &K) const {

        _k->gram(dirX, dirY, X, Y, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<double> &X, El::Matrix<double> &K) const {

        _k->symmetric_gram(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::Matrix<float> &X, El::Matrix<float> &K) const {

        _k->symmetric_gram(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<double> &X,
        El::ElementalMatrix<double> &K) const {

        _k->symmetric_gram(uplo, dir, X, K);
    }

    virtual void symmetric_gram(El::UpperOrLower uplo, base::direction_t dir,
        const El::ElementalMatrix<float> &X,
        El::ElementalMatrix<float> &K) const {

        _k->symmetric_gram(uplo, dir, X, K);
    }

    virtual
    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
        regular_feature_transform_tag tag, base::context_t& context) const {

        return _k->create_rft(S, tag, context);
    }

    virtual
    sketch::sketch_transform_t<boost::any, boost::any> *create_rft(El::Int S,
        fast_feature_transform_tag tag, base::context_t& context) const {

        return _k->create_rft(S, tag, context);
    }

    template<typename IT, typename OT, typename TT>
    sketch::sketch_transform_t<IT, OT> *create_rft(El::Int S,
        TT tag, base::context_t& context) const {

        sketch::generic_sketch_transform_ptr_t p(create_rft(S, tag, context));
        return new sketch::sketch_transform_container_t<IT, OT>(p);
    }

    virtual boost::property_tree::ptree to_ptree() const {

        return _k->to_ptree();
    }

private:
    std::shared_ptr<kernel_t> _k;
    std::string _type;
};

} } 

#endif // SKYLARK_KERNELS_HPP
