#ifndef SKYLARK_KRYLOV_INTERNAL_HPP
#define SKYLARK_KRYLOV_INTERNAL_HPP

#include "../../base/base.hpp"
#include "../../utility/elem_extender.hpp"
#include "../../utility/typer.hpp"

namespace skylark { namespace algorithms {

namespace internal {

template<typename T>
struct scalar_cont_typer_t {

};

template<typename F>
struct scalar_cont_typer_t<El::Matrix<F> > {
    typedef El::Matrix<F> type;

    static type build_compatible(int m, int n, const El::Matrix<F>& A) {
        return type(m, n);
    }
};

template<typename F, El::Distribution U, El::Distribution V>
struct scalar_cont_typer_t<El::DistMatrix<F, U, V> > {
    typedef El::DistMatrix<F, El::STAR, El::STAR> type;

    static type build_compatible(int m, int n, const El::DistMatrix<F, U, V>& A) {
        return type(m, n, A.Grid(), A.Root());
    }
};

template<typename F>
struct scalar_cont_typer_t<base::sparse_matrix_t<F> > {
    typedef El::Matrix<F> type;

    static type build_compatible(int m, int n, const base::sparse_matrix_t<F>& A) {
        return type(m, n);
    }
};

} // namespace internal

} } // namespace skylark::algorithms

#endif // SKYLARK_KRYLOV_INTERNAL_HPP
