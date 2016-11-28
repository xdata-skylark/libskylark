#ifndef SKYLARK_KERNEL_GRAM_HPP
#define SKYLARK_KERNEL_GRAM_HPP

#ifndef SKYLARK_KERNELS_HPP
#error "Include top-level kernels.hpp instead of including individuals headers"
#endif

#include "BaseKernel.hpp"
#include "../../sketch/sketch.hpp"
#include "../feature_transform_tags.hpp"

namespace skylark { namespace ml {

// TODO(Jordi): Write a comment for this function
template<typename Kernel, typename XT, typename YT, typename KT>
void Gram(base::direction_t dirX, base::direction_t dirY,
    const Kernel& k, const XT &X, const YT &Y, KT &K) {

    k.gram(dirX, dirY, X, Y, K);
}

// TODO(Jordi): Write a comment for this function
template<typename Kernel, typename XT, typename KT>
void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const Kernel& k, const XT &X, KT &K) {

    k.symmetric_gram(uplo, dir, X, K);
}

// TODO(Jordi): Write a comment for this function
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

// TODO(Jordi): Write a comment for this function
void SymmetricGram(El::UpperOrLower uplo, base::direction_t dir,
    const kernel_t& k, const boost::any &X, const boost::any &K) {

    SKYLARK_THROW_EXCEPTION (
        base::ml_exception()
          << base::error_msg(
           "SymmetricGram has not yet been implemented for boost::any params"));
}

} }  // skylark::ml

#endif // SKYLARK_KERNELS_HPP
