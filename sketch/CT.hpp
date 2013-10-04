#ifndef CT_HPP
#define CT_HPP

#include "dense_transform.hpp"

namespace skylark {
namespace sketch {

namespace bstrand = boost::random;

/**
 * Cauchy Transform
 *
 * The CT is simply a dense random matrix with i.i.d Cauchy variables
 */
template < typename InputMatrixType,
           typename OutputMatrixType>
struct CT_t :
   public dense_transform_t<InputMatrixType, OutputMatrixType,
                            bstrand::cauchy_distribution > {

    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
                               bstrand::cauchy_distribution > Base;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    CT_t(int N, int S, double C, skylark::sketch::context_t& context)
        : Base(N, S, context) {
        Base::scale = C / static_cast<double>(S);
    }
};

} // namespace sketch
} // namespace skylark

#endif // CT_HPP
