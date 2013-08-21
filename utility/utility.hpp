#ifndef UTILITY_HPP
#define UTILITY_HPP

/**
 * In general, please only include files that you actually want. However, the
 * popular trend is to create behemoth include files that include all else. So,
 * to please people who like this, we are creating one top-level file in each
 * directory.
 */

#include "matrix.hpp"
#include "mapper.hpp"
#include "partitioner.hpp"
#include "reader.hpp"
#include "comm.hpp"
#include "distributions.hpp"

/** 
 * This file contains all the functions that we wrote to support external 
 * software such as Elemental and CombBLAS. These should technically be in
 * the respective softwares.
 */ 
#include "external/external.hpp"

#endif // UTILITY_HPP
