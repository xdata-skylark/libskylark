#ifndef SKYLARK_FUT_HPP
#define SKYLARK_FUT_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif


namespace skylark { namespace sketch {

template<typename ValueType>
struct fft_futs {

};


#include "../utility/fft/fft_futs.h"

#if SKYLARK_HAVE_SPIRALWHT

extern "C" {
#include <spiral_wht.h>
}

template<typename ValueType>
struct WHT_t {

};

template<>
struct WHT_t<double> {

    typedef double value_type;

    WHT_t<double>(int N) : _N(N) {
        // TODO check that N is a power of two.
        _tree = wht_get_tree(ceil(log(N) / log(2)));
    }

    ~WHT_t<double>() {
        delete _tree;
    }

    template <typename Dimension>
    void apply(El::Matrix<value_type>& A, Dimension dimension) const {
        return apply_impl (A, dimension);
    }

    template <typename Dimension>
    void apply_inverse(El::Matrix<value_type>& A, Dimension dimension) const {
        return apply_inverse_impl (A, dimension);
    }

    double scale() const {
        return 1 / sqrt(_N);
    }

private:

    void apply_impl(El::Matrix<value_type>& A,
                    skylark::sketch::columnwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, 1, AA + j * A.LDim());
    }

    void apply_inverse_impl(El::Matrix<value_type>& A,
        skylark::sketch::columnwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, 1, AA + j * A.LDim());
    }

    void apply_impl(El::Matrix<value_type>& A,
        skylark::sketch::rowwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, A.Height(), AA + j);
        // Not sure stride is used correctly here.
    }

    void apply_inverse_impl(El::Matrix<value_type>& A,
        skylark::sketch::rowwise_tag) const {
        ValueType* AA = A.Buffer();
        int j;

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for private(j)
#       endif
        for (j = 0; j < A.Width(); j++)
            wht_apply(_tree, A.Height(), AA + j);
        // Not sure stride is used correctly here.
    }


    const int _N;
    Wht *_tree;
};


#endif // SKYLARK_HAVE_SPIRALWHT

} } /** namespace skylark::sketch */

#endif // SKYLARK_FUT_HPP
