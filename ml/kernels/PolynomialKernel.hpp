#ifndef SKYLARK_POLYNOMIAL_KERNEL_HPP
#define SKYLARK_POLYNOMIAL_KERNEL_HPP

#ifndef SKYLARK_KERNELS_HPP
#error "Include top-level kernels.hpp instead of including individuals headers"
#endif

#include "BaseKernel.hpp"
#include "gram.hpp"
#include "../../sketch/sketch.hpp"
#include "../feature_transform_tags.hpp"

namespace skylark { namespace ml {

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

} }  // skylark::ml

#endif
