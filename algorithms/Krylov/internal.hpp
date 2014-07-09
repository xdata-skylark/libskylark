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
struct scalar_cont_typer_t<elem::Matrix<F> > {
    typedef elem::Matrix<F> type;

    static type build_compatible(int m, int n, const elem::Matrix<F>& A) {
        return type(m, n);
    }
};

template<typename F, elem::Distribution U, elem::Distribution V>
struct scalar_cont_typer_t<elem::DistMatrix<F, U, V> > {
    typedef elem::DistMatrix<F, elem::STAR, elem::STAR> type;

    static type build_compatible(int m, int n, const elem::DistMatrix<F, U, V>& A) {
        return type(m, n, A.Grid(), A.Root());
    }
};

template<typename F>
struct scalar_cont_typer_t<base::sparse_matrix_t<F> > {
    typedef elem::Matrix<F> type;

    static type build_compatible(int m, int n, const base::sparse_matrix_t<F>& A) {
        return type(m, n);
    }
};

} // namespace internal

} } // namespace skylark::algorithms

#endif // SKYLARK_KRYLOV_INTERNAL_HPP
