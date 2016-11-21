#ifndef SKYLARK_BASE_KERNEL_HPP
#define SKYLARK_BASE_KERNEL_HPP

#ifndef SKYLARK_KERNELS_HPP
#error "Include top-level kernels.hpp instead of including individuals headers"
#endif

#include "../../sketch/sketch.hpp"
#include "../feature_transform_tags.hpp"

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

} }  // skylark::ml

#endif // SKYLARK_KERNELS_HPP
