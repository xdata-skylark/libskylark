#ifndef SKYLARK_NLA_HPP
#define SKYLARK_NLA_HPP

/** Include SVD for Elemental ... The guard is here merely as a protection */
#include "sketched_svd.hpp"

/** Include the iterative solver parameters */
#include "iter_params.hpp"

/** Include the iterative solver operator */
#include "iter_solver_op.hpp"

/** Include LSQR. No need for a guard as the guards are all within the files */
#include "lsqr.hpp"

#endif /* SKYLARK_NLA_HPP */
