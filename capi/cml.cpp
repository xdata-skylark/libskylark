#include <El.h>

#include "mlc.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "matrix_types.hpp"
#include "../ml/ml.hpp"

skylark::base::context_t &dref_context(sl_context_t *ctxt);

#include "ckernel.cpp"
