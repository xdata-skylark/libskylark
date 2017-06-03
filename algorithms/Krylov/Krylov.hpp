#ifndef SKYLARK_KRYLOV_HPP
#define SKYLARK_KRYLOV_HPP

namespace skylark { namespace algorithms {

enum krylov_method_t {
    CG_TAG = 0,
    FCG_TAG = 1,
    LSQR_TAG = 2,
    GMRES_TAG = 3
};

} }

#include "krylov_iter_params.hpp"
#include "CG.hpp"
#include "FlexibleCG.hpp"
#include "LSQR.hpp"
#include "Chebyshev.hpp"

#endif
