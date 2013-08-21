#ifndef SKYLARK_SKETCHED_SVD_HPP
#define SKYLARK_SKETCHED_SVD_HPP

#include "../config.h"

namespace skylark { namespace nla {

/** General template */
/**
 * These are the parameters that are needed for sketching SVD.
 * (1) The sketching type to use.
 * (2) The number of singular vectors of the sketched matrix to use.
 *
 * Use:
 * [U_k, S_k, V_k] ~= sketched_SVD (A, k) 
 */ 
template <typename InputMatrixType,
          typename SketchedMatrixType,  
          typename OutputMatrixType,
          template <typename, typename> class SketchTransformType>
struct sketched_svd_t {};

} } /** namespace skylark::nla */

#if SKYLARK_HAVE_ELEMENTAL
#include "sketched_svd_Elemental.hpp"
#endif

#if 0
/** FIXME: Don't know if we will be providing something for CombBLAS */
#if SKYLARK_HAVE_COMBBLAS
#include "sketched_svd_CombBLAS.hpp"
#endif
#endif

#endif /** SKYLARK_SVD_HPP */
