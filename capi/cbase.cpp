#include "matrix_types.hpp"
#include "basec.hpp"
#include "../base/context.hpp"
#include "../base/exception.hpp"

static boost::exception_ptr lastexception;

struct sl_context_t : public skylark::base::context_t {
    sl_context_t(int seed) : skylark::base::context_t(seed) {

   }
};

skylark::base::context_t &dref_context(sl_context_t *ctxt) {
    return *ctxt;
}

extern "C" {

SKYLARK_EXTERN_API const char* sl_strerror(const int error_code) {
    return skylark_strerror(error_code);
}

SKYLARK_EXTERN_API bool sl_has_elemental() {
    return true;
}

SKYLARK_EXTERN_API bool sl_has_combblas() {
#if SKYLARK_HAVE_COMBBLAS
    return true;
#else
    return false;
#endif
}

/* Support for skylark::sl_context_t. */
SKYLARK_EXTERN_API int sl_create_default_context(int seed,
        sl_context_t **ctxt) {
    SKYLARK_BEGIN_TRY()
        *ctxt = new sl_context_t(seed);
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);
    return 0;
}

SKYLARK_EXTERN_API int sl_create_context(int seed,
        MPI_Comm comm, sl_context_t **ctxt) {
    SKYLARK_BEGIN_TRY()
        *ctxt = new sl_context_t(seed);
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);
    return 0;
}

SKYLARK_EXTERN_API int sl_free_context(sl_context_t *ctxt) {
    SKYLARK_BEGIN_TRY()
        delete ctxt;
    SKYLARK_END_TRY()
    SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);
    return 0;
}

SKYLARK_EXTERN_API int sl_wrap_raw_matrix(double *data, int m, int n, void **A)
{
    Matrix *tmp = new Matrix();
    tmp->Attach(m, n, data, m);
    *A = tmp;
    return 0;
}

SKYLARK_EXTERN_API int sl_free_raw_matrix_wrap(void *A_) {
    delete static_cast<Matrix *>(A_);
    return 0;
}


SKYLARK_EXTERN_API int sl_wrap_raw_sp_matrix(int *indptr, int *ind, double *data,
    int nnz, int n_rows, int n_cols, void **A)
{
    SparseMatrix *tmp = new SparseMatrix();
    tmp->attach(indptr, ind, data, nnz, n_rows, n_cols);
    *A = tmp;
    return 0;
}

SKYLARK_EXTERN_API int sl_free_raw_sp_matrix_wrap(void *A_) {
    delete static_cast<SparseMatrix *>(A_);
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_struct_updated(void *A_,
        bool *struct_updated) {
    *struct_updated = static_cast<SparseMatrix *>(A_)->struct_updated();
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_reset_update_flag(void *A_) {
    static_cast<SparseMatrix *>(A_)->reset_update_flag();
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_nnz(void *A_, int *nnz) {
    *nnz = static_cast<SparseMatrix *>(A_)->nonzeros();
    return 0;
}

SKYLARK_EXTERN_API int sl_raw_sp_matrix_data(void *A_, int32_t *indptr,
        int32_t *indices, double *values) {
    static_cast<SparseMatrix *>(A_)->detach(indptr, indices, values);
    return 0;
}

SKYLARK_EXTERN_API void sl_get_exception_info(char **info) {
    std::string infos = boost::diagnostic_information(lastexception);
    *info = new char[infos.length() + 1];
    std::strcpy(*info, infos.c_str());
}

SKYLARK_EXTERN_API void sl_print_exception_trace() {
    try {
        boost::rethrow_exception(lastexception);
    } catch (const skylark::base::skylark_exception &ex) {
        SKYLARK_PRINT_EXCEPTION_TRACE(ex);
    }
}

} // extern "C"
