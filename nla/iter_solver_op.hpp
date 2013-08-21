#ifndef SKYLARK_ITER_SOLVER_OP_HPP
#define SKYLARK_ITER_SOLVER_OP_HPP

#include "../config.h"

namespace skylark { namespace nla {

template <typename MatrixType,
          typename MultiVectorType>
struct iter_solver_op_t { };

} } /** namespace skylark::nla */

#if SKYLARK_HAVE_ELEMENTAL
#include "iter_solver_op_Elemental.hpp"
#endif

#if SKYLARK_HAVE_COMBBLAS
#include "iter_solver_op_CombBLAS.hpp"
#endif

#endif // SKYLARK_ITER_SOLVER_OP_HPP
