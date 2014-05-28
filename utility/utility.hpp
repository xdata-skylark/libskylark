#ifndef SKYLARK_UTILITY_HPP
#define SKYLARK_UTILITY_HPP

/**
 * In general, please only include files that you actually want. However, the
 * popular trend is to create behemoth include files that include all else. So,
 * to please people who like this, we are creating one top-level file in each
 * directory.
 */

#include "comm.hpp"
#include "distributions.hpp"
#include "randgen.hpp"
#include "get_communicator.hpp"
#include "typer.hpp"
#include "elem_extender.hpp"

/**
 * This file contains all the functions that we wrote to support external 
 * software such as Elemental and CombBLAS. These should technically be in
 * the respective softwares.
 */
#include "external/external.hpp"

#endif // UTILITY_HPP
