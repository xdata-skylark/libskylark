#ifndef SKYLARK_KERNEL_CONTAINER_HPP
#define SKYLARK_KERNEL_CONTAINER_HPP

#ifndef SKYLARK_KERNELS_HPP
#error "Include top-level kernels.hpp instead of including individuals headers"
#endif

#include "BaseKernel.hpp"
#include "LinearKernel.hpp"
#include "MaternKernel.hpp"
#include "GaussianKernel.hpp"
#include "LaplacianKernel.hpp"
#include "PolynomialKernel.hpp"
#include "ExpsemigroupKernel.hpp"

namespace skylark { namespace ml {

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
};

} }  // skylark::ml

#endif
