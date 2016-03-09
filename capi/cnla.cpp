#include "nlac.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

// TODO for now... but we can use the any,any to implement c-api
#define SKYLARK_NO_ANY
#include "matrix_types.hpp"
#include "sketchc.hpp"
#include "../nla/nla.hpp"

namespace nla = skylark::nla;

skylark::base::context_t &dref_context(sl_context_t *ctxt);

extern "C" {

SKYLARK_EXTERN_API int sl_approximate_symmetric_svd(
    char *A_type, void *A_, char *S_type, void *S_, char *V_type, void *V_,
    uint16_t k, char *params, sl_context_t *ctxt) {

    //lower ? El::LOWER : El:UPPER
    boost::property_tree::ptree json_tree;
    std::stringstream data;
    data << params;
    boost::property_tree::read_json(data, json_tree);
    nla::approximate_svd_params_t parms(json_tree);

    matrix_type_t A_t = str2matrix_type(A_type);
    matrix_type_t S_t = str2matrix_type(S_type);
    matrix_type_t V_t = str2matrix_type(V_type);

# define AUTO_SYM_SVD_DISPATCH(At, St, Vt, AT, ST, VT)                   \
    if (A_t == At && S_t == St && V_t == Vt) {                           \
        AT &A = * static_cast<AT*>(A_);                                  \
        ST &S = * static_cast<ST*>(S_);                                  \
        VT &V = * static_cast<VT*>(V_);                                  \
                                                                         \
        SKYLARK_BEGIN_TRY()                                              \
            skylark::nla::ApproximateSymmetricSVD(El::LOWER, A, V, S, k, \
                dref_context(ctxt), parms);                              \
        SKYLARK_END_TRY()                                                \
        SKYLARK_CATCH_AND_RETURN_ERROR_CODE();                           \
    }

    AUTO_SYM_SVD_DISPATCH(
        MATRIX, MATRIX, MATRIX,
        Matrix, Matrix, Matrix);

    AUTO_SYM_SVD_DISPATCH(
        DIST_MATRIX, DIST_MATRIX, DIST_MATRIX,
        DistMatrix, DistMatrix, DistMatrix);

    return 0;
}

SKYLARK_EXTERN_API int sl_approximate_svd(
    char *A_type, void *A_, char *U_type, void *U_,
    char *S_type, void *S_, char *V_type, void *V_,
    uint16_t k, char *params, sl_context_t *ctxt) {

    boost::property_tree::ptree json_tree;
    std::stringstream data;
    data << params;
    boost::property_tree::read_json(data, json_tree);
    nla::approximate_svd_params_t parms(json_tree);

    matrix_type_t A_t = str2matrix_type(A_type);
    matrix_type_t U_t = str2matrix_type(U_type);
    matrix_type_t S_t = str2matrix_type(S_type);
    matrix_type_t V_t = str2matrix_type(V_type);

# define AUTO_SVD_DISPATCH(At, Ut, St, Vt, AT, UT, ST, VT)               \
    if (A_t == At && U_t == Ut && S_t == St && V_t == Vt) {              \
        AT &A = * static_cast<AT*>(A_);                                  \
        UT &U = * static_cast<UT*>(U_);                                  \
        ST &S = * static_cast<ST*>(S_);                                  \
        VT &V = * static_cast<VT*>(V_);                                  \
                                                                         \
        SKYLARK_BEGIN_TRY()                                              \
            skylark::nla::ApproximateSVD(A, U, S, V, k,                  \
                dref_context(ctxt), parms);                              \
        SKYLARK_END_TRY()                                                \
        SKYLARK_CATCH_AND_RETURN_ERROR_CODE();                           \
    }

    AUTO_SVD_DISPATCH(
        MATRIX, MATRIX, MATRIX, MATRIX,
        Matrix, Matrix, Matrix, Matrix);

    AUTO_SVD_DISPATCH(
        DIST_MATRIX, DIST_MATRIX,
        DIST_MATRIX, DIST_MATRIX,
        DistMatrix, DistMatrix, DistMatrix, DistMatrix);

    return 0;
}

} // extern "C"
