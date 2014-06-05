#ifndef SKYLARK_PRECOND_HPP
#define SKYLARK_PRECOND_HPP

#include "../../base/base.hpp"

namespace skylark { namespace nla {

/**
 * Defines a generic interface for preconditioners.
 *
 * @tparam T type of solution vector on which the preconditioner is applied.
 */
template<typename T>
struct precond_t {

    /// Apply the preconditoiner to X, overwritting X.
    virtual void apply(T& X) const = 0;

    virtual void apply_adjoint(T& X) const = 0;

    virtual ~precond_t() { }
};

#ifdef SKYLARK_HAVE_ELEMENTAL

/**
 * Identity preconditioner - does nothing!
 */
template<typename T>
struct id_precond_t : public precond_t<T> {
    void apply(T& X) const { }

    void apply_adjoint(T& X) const { }
};

/**
 * A preconditioner that is explicitly given as a matrix.
 */
template<typename XType, typename NType>
struct mat_precond_t : public precond_t<XType> {
    const NType& N;

    mat_precond_t(const NType& N) : N(N) { }

    void apply(XType& X) const {
        XType Xin(X);
        base::Gemm(elem::NORMAL, elem::NORMAL, 1.0, N, Xin, X);
    }

    void apply_adjoint(XType& X) const {
        XType Xin(X);
        base::Gemm(elem::ADJOINT, elem::NORMAL, 1.0, N, Xin, X);
    }
};

/**
 * A preconditioner that is the inverse of a given traingular matrix.
 */
template<typename XType, typename RType,
         elem::UpperOrLower UL, elem::UnitOrNonUnit D>
struct tri_inverse_precond_t : public precond_t<XType> {
    const RType& R;

    tri_inverse_precond_t(const RType& R) : R(R) { }

    void apply(XType& X) const {
        base::Trsm(elem::LEFT, UL, elem::NORMAL, D, 1.0, R, X);
    }

    void apply_adjoint(XType& X) const {
        base::Trsm(elem::LEFT, UL, elem::ADJOINT, D, 1.0, R, X);
    }
};

#endif

} } /** namespace skylark::nla */

#endif // SKYLARK_PRECOND_HPP
