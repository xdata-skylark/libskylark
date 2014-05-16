#ifndef SKYLARK_COMPUTED_MATRIX_HPP
#define SKYLARK_COMPUTED_MATRIX_HPP

#include "../utility/typer.hpp"

namespace skylark { namespace base {

/**
 * Defines an interface that allows us to work with computed matrices,
 * i.e. matrices can be materilized as needed. These kind of matrices are
 * attractive if the information needed to materilize the matrix is much
 * smaller than the materilized matrix, and the materilization is
 * relative inexpensive. In that case, using a computed matrix can reduce the
 * memory footprint quite a lot.
 */
template<typename MatrixType>
struct computed_matrix_t {
    typedef MatrixType materialized_type;
    typedef typename utility::typer_t<MatrixType>::index_type index_type;

    virtual index_type height() const = 0;
    virtual index_type width() const = 0;
    virtual void materialize(materialized_type &Z) const = 0;
    virtual materialized_type materialize() const = 0;
};

} } // namespace skylark::base

#endif // SKYLARK_COMPUTED_MATRIX_HPP
