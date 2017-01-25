#ifndef SKYLARK_EXPSEMIGROUP_KERNEL_HPP
#define SKYLARK_EXPSEMIGROUP_KERNEL_HPP

#ifndef SKYLARK_KERNELS_HPP
#error "Include top-level kernels.hpp instead of including individuals headers"
#endif

#include "BaseKernel.hpp"
#include "gram.hpp"
#include "../../sketch/sketch.hpp"
#include "../feature_transform_tags.hpp"

namespace skylark { namespace ml {

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
        El::EntrywiseMap(K, std::function<value_type(const value_type &)> (
            [this] (const value_type &x) {
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

} }  // skylark::ml

#endif
