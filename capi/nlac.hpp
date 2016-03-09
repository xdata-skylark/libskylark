#ifndef SKYLARK_NLAC_HPP
#define SKYLARK_NLAC_HPP

#include "mpi.h"
#include "config.h"

#include "basec.hpp"

extern "C" {

/**
 * Approximate symmetric SVD (A = V S V^T).
 *
 * \param A_type type of the input matrix
 * \param A_ input matrix
 * \param S_type type of the matrix holding the singular values
 * \param S_ singular values
 * \param V_type type of the right singular vectors
 * \param V_ right singular vectors
 * \param k target rank
 * \param params_json json serialized approximate_svd_params_t
 * \param ctxt libSkylark context
 */
SKYLARK_EXTERN_API int sl_approximate_symmetric_svd(
    char *A_type, void *A_, char *S_type, void *S_, char *V_type, void *V_,
    uint16_t k, char *params_json, sl_context_t *ctxt);

/**
 * Approximate SVD (A = U S V^T).
 *
 * \param A_type type of the input matrix
 * \param A_ input matrix
 * \param U_type type of the left singular vectors
 * \param U_ left singular vectors
 * \param S_type type of the matrix holding the singular values
 * \param S_ singular values
 * \param V_type type of the right singular vectors
 * \param V_ right singular vectors
 * \param k target rank
 * \param params_json json serialized approximate_svd_params_t
 * \param ctxt libSkylark context
 */
SKYLARK_EXTERN_API int sl_approximate_svd(
    char *A_type, void *A_, char *U_type, void *U_,
    char *S_type, void *S_, char* V_type, void *V_,
    uint16_t k, char *params_json, sl_context_t *ctxt);

} // extern "C"

#endif // SKYLARK_NLAC_HPP
