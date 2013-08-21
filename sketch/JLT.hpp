#ifndef JLT_HPP
#define JLT_HPP

#include "denset.hpp"

namespace skylark {
namespace sketch {

namespace bstrand = boost::random;

/**
 * Johnson-Lindenstrauss Transform
 *
 * The JLT is simply a dense random matrix with i.i.d normal entries.
 */
template < typename InputMatrixType,
           typename OutputMatrixType>
struct JLT_t :
   public dense_transform_t<InputMatrixType, OutputMatrixType,
                            bstrand::normal_distribution > {

    typedef dense_transform_t<InputMatrixType, OutputMatrixType,
                               bstrand::normal_distribution > Base;
    /**
     * Constructor
     * Most of the work is done by base. Here just write scale
     */
    JLT_t(int N, int S, skylark::sketch::context_t& context)
        : Base(N, S, context) {
        Base::scale = sqrt(1.0 / static_cast<double>(S));
    }
};

} // namespace sketch
} // namespace skylark

#endif // JLT_HPP
