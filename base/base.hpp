#ifndef SKYLARK_BASE_HPP
#define SKYLARK_BASE_HPP

#include "../config.h"

#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif

#if SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif

#include "exception.hpp"
#include "sparse_matrix.hpp"
#include "computed_matrix.hpp"
#include "basic.hpp"
#include "query.hpp"
#include "viewing.hpp"
#include "copy.hpp"
#include "Trsm.hpp"
#include "Gemm.hpp"
#include "Gemv.hpp"
#include "inner.hpp"
#include "svd.hpp"
#include "context.hpp"

#endif // SKYLARK_BASE_HPP
