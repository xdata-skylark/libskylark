#ifndef SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_HPP
#define SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_HPP

#include "fast_exact_regressor.hpp"

namespace skylark {
namespace algorithms {

// Base class for linear L2 fast algorithms.
struct linearl2_reg_fast_alg_tag { };

// Simplified Blendenpik just does a single sketch and uses the
// sketched matrix as a preconditioner.
template<template <typename, typename> class TransformType>
struct simplified_blendenpik_tag : public linearl2_reg_fast_alg_tag { };

// The algorithm described in the Blendenpik paper
struct blendenpik_tag : public linearl2_reg_fast_alg_tag { };

// The algorithm described in the LSRN paper
struct lsrn_tag : public linearl2_reg_fast_alg_tag { };

} }

#if SKYLARK_HAVE_ELEMENTAL
#include "fast_linearl2_exact_regressor_Elemental.hpp"
#endif

#endif // SKYLARK_FAST_LINEARL2_EXACT_REGRESSOR_HPP
