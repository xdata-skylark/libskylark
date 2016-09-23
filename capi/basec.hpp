#ifndef SKYLARK_BASEC_HPP
#define SKYLARK_BASEC_HPP

// Some tools require special API declaration. Customizing the
// SKYLARK_EXTERN_API allows this. The default is simply nothing.
#ifndef SKYLARK_EXTERN_API
#define SKYLARK_EXTERN_API
#endif

struct sl_context_t;

#define SL_COLUMNWISE 1
#define SL_ROWWISE    2

#define SL_COLUMNS 1
#define SL_ROWS    2

extern "C" {

/** Returns a string describing the sketch transforms that are supported.
 */
SKYLARK_EXTERN_API char *sl_supported_sketch_transforms();

/** Converting an error code to a human readable string describing the failure
 *  @param errorcode the error code to resolve
 *  @return a string containing the error message
 */
SKYLARK_EXTERN_API const char* sl_strerror(const int errorcode);

/** Provide mechanism to check for add-ons.
 *  @return a boolean if the add-on library is enabled
 */
SKYLARK_EXTERN_API bool sl_has_elemental();
SKYLARK_EXTERN_API bool sl_has_combblas();

// Support for sl_context_t.

/** Creating a default Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @param ctxt Skylark context
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API int sl_create_default_context(int seed,
    sl_context_t **ctxt);

/** Creating a Skylark context required for applying sketches.
 *  @param seed for the rng generator
 *  @param cm MPI communicator used
 *  @param ctxt Skylark context
 *  @return a Skylark context
 */
SKYLARK_EXTERN_API int sl_create_context(int seed, MPI_Comm cm,
    sl_context_t **ctxt);

/** Free resources hold by a Skylark context.
 *  @param ctxt Skylark context
 */
SKYLARK_EXTERN_API int sl_free_context(sl_context_t *ctxt);

// Helper functions to allow wrapping of object

SKYLARK_EXTERN_API int sl_wrap_raw_matrix(double *data, int m, int n, void **A);

SKYLARK_EXTERN_API int sl_free_raw_matrix_wrap(void *A_);

SKYLARK_EXTERN_API int sl_wrap_raw_sp_matrix(int *indptr, int *ind,
    double *data, int nnz, int n_rows, int n_cols, void **A);

SKYLARK_EXTERN_API int sl_free_raw_sp_matrix_wrap(void *A_);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_struct_updated(void *A_,
    bool *struct_updated);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_reset_update_flag(void *A_);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_nnz(void *A_, int *nnz);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_height(void *A_, int *height);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_width(void *A_, int *width);

SKYLARK_EXTERN_API int sl_raw_sp_matrix_data(void *A_, int32_t *indptr,
        int32_t *indices, double *values);

SKYLARK_EXTERN_API void sl_get_exception_info(char **info);

SKYLARK_EXTERN_API void sl_print_exception_trace();

} // extern "C"

#endif // SKYLARK_BASEC_HPP
