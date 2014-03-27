#ifndef SKYLARK_SKETCH_HPP
#define SKYLARK_SKETCH_HPP

/**
 * In general, please only include files that you actually want. However, the
 * popular trend is to create behemoth include files that include all else. So,
 * to please people who like this, we are creating one top-level file in each
 * directory.
 */
#include "transforms.hpp"
#include "FUT.hpp"
#include "RFUT.hpp"
#include "dense_transform.hpp"
#include "CT.hpp"
#include "JLT.hpp"
#include "FJLT.hpp"
#include "RFT.hpp"
#include "FRFT.hpp"
#include "RLT.hpp"
#include "CWT.hpp"
#include "MMT.hpp"
#include "WZT.hpp"
#include "PPT.hpp"

/** skylark::base::context_t also happens to be a powerful utility for RNG*/
#include "../base/context.hpp"

#endif // SKYLARK_SKETCH_HPP
