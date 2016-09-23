#include <El.h>

#include "ioc.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "matrix_types.hpp"
#include "../utility/utility.hpp"

extern "C" {

SKYLARK_EXTERN_API int sl_readlibsvm(char *fname,
    char *X_type, void *X_, char *Y_type, void *Y_,
    int direction_, int min_d, int max_n) {

    skylark::base::direction_t direction =
        direction_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;

    SKYLARK_BEGIN_TRY()
        skylark::utility::io::ReadLIBSVM(fname,
            skylark_void2any(X_type, X_),
            skylark_void2any(Y_type, Y_),
            direction, min_d, max_n);
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    return 0;

}

} // extern "C"
