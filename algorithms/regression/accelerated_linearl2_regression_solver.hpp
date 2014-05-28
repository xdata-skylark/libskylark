#ifndef SKYLARK_ACCELERATED_LINEARL2_REGRESSION_SOLVER_HPP
#define SKYLARK_ACCELERATED_LINEARL2_REGRESSION_SOLVER_HPP

#include "accelerated_regression_solver.hpp"

namespace skylark {
namespace algorithms {

//****** Tags for how to build the preconditioner (QR or SVD)
struct precond_alg_tag { };

struct qr_precond_tag : precond_alg_tag { };

struct svd_precond_tag : precond_alg_tag { };

//****** Tags for algorithm for fast linear L2 regresssion.
struct linearl2_reg_fast_alg_tag { };

// Simplified Blendenpik just does a single sketch and uses the
// sketched matrix as a preconditioner.
template<template <typename, typename> class TransformType, 
         typename PrecondTag = qr_precond_tag>
struct simplified_blendenpik_tag : public linearl2_reg_fast_alg_tag { };

// The algorithm described in the Blendenpik paper
template<typename PrecondTag = qr_precond_tag>
struct blendenpik_tag : public linearl2_reg_fast_alg_tag { };

// The algorithm described in the LSRN paper
template<typename PrecondTag = svd_precond_tag>
struct lsrn_tag : public linearl2_reg_fast_alg_tag { };

} }

#if SKYLARK_HAVE_ELEMENTAL
#include "accelerated_linearl2_regression_solver_Elemental.hpp"
#endif

#endif // SKYLARK_ACCELERATED_LINEARL2_REGRESSION_SOLVER_HPP
