#ifndef SKYLARK_MATERN_KERNEL_HPP
#define SKYLARK_MATERN_KERNEL_HPP

#ifndef SKYLARK_KERNELS_HPP
#error "Include top-level kernels.hpp instead of including individuals headers"
#endif

#include "BaseKernel.hpp"
#include "gram.hpp"
#include "../../sketch/sketch.hpp"
#include "../feature_transform_tags.hpp"

namespace skylark { namespace ml {

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

} }  // skylark::ml

#endif
