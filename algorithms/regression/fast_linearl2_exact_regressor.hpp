#ifndef SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_HPP
#define SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_HPP

#include "fast_exact_regressor.hpp"

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
#include "fast_linearl2_exact_regressor_Elemental.hpp"
#endif

#endif // SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_HPP
