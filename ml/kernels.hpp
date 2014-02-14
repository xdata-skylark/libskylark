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

} } }

#endif // SKYLARK_KERNELS_HPP
