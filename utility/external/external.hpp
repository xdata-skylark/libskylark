#ifndef SKYLARK_EXTERNAL_HPP
#define SKYLARK_EXTERNAL_HPP

/** Include a version of FullyDistMultiVec */
#ifdef SKYLARK_HAVE_COMBBLAS
#include "FullyDistMultiVec.hpp"
#endif /** SKYLARK_HAVE_COMBBLAS */

/** Include the printing function */
#include "print.hpp"

#endif /** SKYLARK_EXTERNAL_HPP */
