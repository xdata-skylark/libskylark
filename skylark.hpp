#ifndef SKYLARK_HPP
#define SKYLARK_HPP

#include <boost/mpi.hpp>
#include <boost/random.hpp>

/*! \mainpage Skylark
 *
 * The immediate goal of this framework is to provide implementations
 * of sketching-based NLA kernels on a distributed platform, and to broaden
 * the classes of matrices for which these specific implementations work
 * well.
 *
 * \section intro_sec Introduction
 *
 * This software stack provides sketching-based NLA kernels for more general
 * data analysis and optimization applications; such tasks have significantly
 * different input matrices and performance criteria than arise in the more
 * traditional scientific computing applications.
 * The crucial NLA kernels to be implemented include regression and low-rank
 * approximations of matrices, akin to the Singular Value Decomposition
 * (SVD).
 *
 * Additionally this library provides a simple distributed Python interface.
 *
 * \section req_sec Requirements
 *
 * The framework requires the following dependencies:
 *
 *   - Boost
 *   - Python
 *   - Elemental
 *   - BLAS/LAPACK
 *   - FFTW (optional, required for fast transformations)
 *
 * See the INSTALL file for build instructions.
*/

/** TODO: Add #ifdef guard */
#if SKYLARK_HAVE_ELEMENTAL
# include <elemental.hpp>
#endif

/** Include all the configuration information */
#include "config.h"

/** Include all the utility functions */
#include "utility/utility.hpp"

/** Include all the sketching primitives */
#include "sketch/sketch.hpp"

#endif // SKYLARK_HPP
