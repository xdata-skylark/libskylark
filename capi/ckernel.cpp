enum kernel_type_t {
    KERNEL_TYPE_ERROR,
    LINEAR,
    POLYNOMIAL,
    GAUSSIAN,
    LAPLACIAN,
    EXPSEMIGROUP,
    MATERN
};

static kernel_type_t str2kernel_type(const char *str) {
    STRCMP_TYPE(linear, LINEAR);
    STRCMP_TYPE(polynomial, POLYNOMIAL);
    STRCMP_TYPE(gaussian, GAUSSIAN);
    STRCMP_TYPE(laplacian, LAPLACIAN);
    STRCMP_TYPE(expsemigroup, EXPSEMIGROUP);
    STRCMP_TYPE(matern, MATERN);

    return KERNEL_TYPE_ERROR;
}

struct sl_kernel_t {
    const kernel_type_t type;
    skylark::ml::kernel_container_t kernel_obj; 

    sl_kernel_t(kernel_type_t type,
        const skylark::ml::kernel_container_t &kernel_obj)
        : type(type), kernel_obj(kernel_obj) {}
};

extern "C" {


SKYLARK_EXTERN_API int sl_create_kernel(
    char *type_, int N, sl_kernel_t **kernel, ...) {

    kernel_type_t type = str2kernel_type(type_);

    if (type == KERNEL_TYPE_ERROR)
        return 111;

    std::shared_ptr<skylark::ml::kernel_t> k_ptr;

    if (type == GAUSSIAN) {
        va_list argp;
        va_start(argp, kernel);
        double sigma = va_arg(argp, double);
        k_ptr.reset(new skylark::ml::gaussian_t(N, sigma));
        va_end(argp);
    }

    if (type == LAPLACIAN) {
        va_list argp;
        va_start(argp, kernel);
        double sigma = va_arg(argp, double);
        k_ptr.reset(new skylark::ml::laplacian_t(N, sigma));
        va_end(argp);
    }

    if (type == POLYNOMIAL) {
        va_list argp;
        va_start(argp, kernel);
        int q = va_arg(argp, int);
        double c = va_arg(argp, double);
        double g = va_arg(argp, double);
        k_ptr.reset(new skylark::ml::polynomial_t(N, q, c, g));
        va_end(argp);
    }

    *kernel = new sl_kernel_t(type, k_ptr);

    return 0;
}

SKYLARK_EXTERN_API
    int sl_free_kernel(sl_kernel_t *k) {

    delete k;

    return 0;
}


SKYLARK_EXTERN_API int sl_kernel_gram(int dirX_, int dirY_,
    sl_kernel_t *k,  char *X_type, void *X_, char *Y_type, void *Y_,
    char *K_type, void *K_) {

    skylark::base::direction_t dirX =
        dirX_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;
    skylark::base::direction_t dirY =
        dirY_ == SL_COLUMNS ? skylark::base::COLUMNS : skylark::base::ROWS;

    SKYLARK_BEGIN_TRY()
        skylark::ml::Gram(dirX, dirY,
            k->kernel_obj, skylark_void2any(X_type, X_),
            skylark_void2any_root(Y_type, Y_),
            skylark_void2any_root(K_type, K_));
    SKYLARK_END_TRY()
        SKYLARK_CATCH_COPY_AND_RETURN_ERROR_CODE(lastexception);

    return 0;
}

} // extern "C"
