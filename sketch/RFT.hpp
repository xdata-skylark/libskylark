#ifndef RFT_HPP
#define RFT_HPP

#include "../config.h"

namespace skylark {
namespace sketch {

namespace bstrand = boost::random;

/**
 * Random Features Transform
 *
 * This the non-linear transform described by Rahimi and Recht.
 * TODO expand this description.
 */
template < typename InputMatrixType,
           typename OutputMatrixType,
           template <typename> class Distribution>
class RFT_t {
    // To be specilized and derived.

};

/**
 * Random Features for Gaussian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType>
struct GaussianRFT_t :
    public RFT_t<InputMatrixType, OutputMatrixType,
                 bstrand::normal_distribution > {

    typedef RFT_t<InputMatrixType, OutputMatrixType,
                  bstrand::normal_distribution > Base;

    GaussianRFT_t(int N, int S, double sigma,
        skylark::sketch::context_t& context)
        : Base(N, S, sigma, context) { };
};

/**
 * Random Features for Laplacian Kernel
 */
template< typename InputMatrixType,
          typename OutputMatrixType>
struct LaplacianRFT_t :
    public RFT_t<InputMatrixType, OutputMatrixType,
                 bstrand::cauchy_distribution > {

    typedef RFT_t<InputMatrixType, OutputMatrixType,
                  bstrand::cauchy_distribution > Base;

    LaplacianRFT_t(int N, int S, double sigma,
        skylark::sketch::context_t& context)
        : Base(N, S, sigma, context) { };
};

} // namespace sketch
} // namespace skylark


#if SKYLARK_HAVE_ELEMENTAL
#include "RFT_Elemental.hpp"
#endif

#endif // RFT_HPP
