#ifndef SKYLARK_IOC_HPP
#define SKYLARK_IOC_HPP

// Some tools require special API declaration. Customizing the
// SKYLARK_EXTERN_API allows this. The default is simply nothing.
#ifndef SKYLARK_EXTERN_API
#define SKYLARK_EXTERN_API
#endif

struct sl_context_t;

extern "C" {

SKYLARK_EXTERN_API int sl_readlibsvm(char *fname,
    char *X_type, void *X_, char *Y_type, void *Y_,
    int direction_, int min_d = 0, int max_n = -1);


} // extern "C"

#endif // SKYLARK_IOC_HPP
