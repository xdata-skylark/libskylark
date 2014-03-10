#ifndef SKYLARK_KERNELS_HPP
#define SKYLARK_KERNELS_HPP

#include "../sketch/sketch.hpp"
#include "feature_transform_tags.hpp"

namespace skylark { namespace ml { namespace kernels {

namespace skysk = skylark::sketch;

struct gaussian_t {

    gaussian_t(int N, double sigma) : _N(N), _sigma(sigma) {

    }

    template<typename IT, typename OT>
    skysk::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, skysk::context_t& context) const {
        return
            new skysk::GaussianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    template<typename IT, typename OT>
    skysk::sketch_transform_t<IT, OT> *create_rft(int S,
        fast_feature_transform_tag, skysk::context_t& context) const {
        return
            new skysk::FastGaussianRFT_t<IT, OT>(_N, S, _sigma, context);
    }

    // TODO method for gram matrix ?


private:
    const int _N;
    const double _sigma;
};

struct polynomial_t {

    polynomial_t(int N, int q = 2, double c = 1.0, double gamma = 1.0)
        : _N(N), _q(q), _c(c), _gamma(gamma) {

    }

    template<typename IT, typename OT>
    skysk::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, skysk::context_t& context) const {
        return
            new skysk::PPT_t<IT, OT>(_N, S, _q, _c, _gamma, context);
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
    skysk::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, skysk::context_t& context) const {
        return
            new skysk::LaplacianRFT_t<IT, OT>(_N, S, _sigma, context);
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
    skysk::sketch_transform_t<IT, OT> *create_rft(int S,
        regular_feature_transform_tag, skysk::context_t& context) const {
        return
            new skysk::ExpSemigroupRLT_t<IT, OT>(_N, S, _beta, context);
    }

    // TODO method for gram matrix ?


private:
    const int _N;
    const double _beta;
};

} } }

#endif // SKYLARK_KERNELS_HPP
