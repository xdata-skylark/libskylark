#ifndef SKYLARK_PRECOND_HPP
#define SKYLARK_PRECOND_HPP

#include "../../base/base.hpp"

namespace skylark { namespace algorithms {

/**
 * Defines a generic interface for inplace preconditioners.
 *
 * @tparam T type of solution vector on which the preconditioner is applied.
 */
template<typename T>
struct inplace_precond_t {

    // Is this really a no-op? (Identity preconditioner)
    virtual bool is_id() const = 0;

    /// Apply the preconditoiner to X, overwritting X.
    virtual void apply(T& X) const = 0;

    virtual void apply_adjoint(T& X) const = 0;

    virtual ~inplace_precond_t() { }
};

/**
 * Defines a generic interface for out-of-place preconditioners.
 *
 * @tparam T type of solution vector on which the preconditioner is applied.
 */
template<typename RhsType, typename SolType>
struct outplace_precond_t {

    // Is this really a no-op? (Identity preconditioner)
    virtual bool is_id() const = 0;

    /// Apply the preconditoiner to X, overwritting X.
    virtual void apply(const RhsType& B, SolType& X) const = 0;

    virtual void apply_adjoint(const RhsType& B, SolType& X) const = 0;

    virtual ~outplace_precond_t() { }
};


/**
 * Inplace identity preconditioner - does nothing!
 */
template<typename T>
struct inplace_id_precond_t : public inplace_precond_t<T> {
    bool is_id() const { return true; }

    void apply(T& X) const { }

    void apply_adjoint(T& X) const { }
};

/**
 * Out-of-place identity preconditioner - just copies.
 */
template<typename RhsType, typename SolType>
struct outplace_id_precond_t : public outplace_precond_t<RhsType, SolType> {
    bool is_id() const { return true; }

    void apply(const RhsType& B, SolType& X) const { base::Copy(B, X); }

    void apply_adjoint(const RhsType& B, SolType& X) const { base::Copy(B, X); }
};

/**
 * A preconditioner that is explicitly given as a matrix.
 */
template<typename XType, typename NType>
struct inplace_mat_precond_t : public inplace_precond_t<XType> {
    const NType& N;

    inplace_mat_precond_t(const NType& N) : N(N) { }

    bool is_id() const { return false; }

    void apply(XType& X) const {
        XType Xin(X);
        typedef typename utility::typer_t<XType>::value_type value_type;
        base::Gemm(El::NORMAL, El::NORMAL, value_type(1.0), N, Xin, X);
    }

    void apply_adjoint(XType& X) const {
        XType Xin(X);
        typedef typename utility::typer_t<XType>::value_type value_type;
        base::Gemm(El::ADJOINT, El::NORMAL, value_type(1.0), N, Xin, X);
    }
};

/**
 * A preconditioner that is the inverse of a given triangular matrix.
 */
template<typename XType, typename RType,
         El::UpperOrLower UL, El::UnitOrNonUnit D>
struct inplace_tri_inverse_precond_t : public inplace_precond_t<XType> {
    const RType& R;

    inplace_tri_inverse_precond_t(const RType& R) : R(R) { }

    bool is_id() const { return false; }

    void apply(XType& X) const {
        typedef typename utility::typer_t<XType>::value_type value_type;
        base::Trsm(El::LEFT, UL, El::NORMAL, D, value_type(1.0), R, X);
    }

    void apply_adjoint(XType& X) const {
        typedef typename utility::typer_t<XType>::value_type value_type;
        base::Trsm(El::LEFT, UL, El::ADJOINT, D, value_type(1.0), R, X);
    }
};

} } /** namespace skylark::algorithms */

#endif // SKYLARK_PRECOND_HPP
