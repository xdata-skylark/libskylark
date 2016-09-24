#ifndef SKYLARK_MATRIX_TYPES_HPP
#define SKYLARK_MATRIX_TYPES_HPP

#include <config.h>
#include <boost/any.hpp>

#include <El.hpp>
#ifdef SKYLARK_HAVE_COMBBLAS
#include <CombBLAS.h>
#endif

#include "../base/sparse_matrix.hpp"

# define STRCMP_TYPE(STR, TYPE) \
    if (std::strcmp(str, #STR) == 0) \
        return TYPE;

// Just for shorter notation
typedef El::Matrix<double> Matrix;
typedef El::ElementalMatrix<double> ElementalMatrix;
typedef El::DistMatrix<double, El::STAR, El::STAR> SharedMatrix;
typedef El::DistMatrix<double, El::CIRC, El::CIRC> RootMatrix;
typedef El::DistMatrix<double> DistMatrix;
typedef El::DistMatrix<double, El::VR, El::STAR> DistMatrix_VR_STAR;
typedef El::DistMatrix<double, El::VC, El::STAR> DistMatrix_VC_STAR;
typedef El::DistMatrix<double, El::STAR, El::VR> DistMatrix_STAR_VR;
typedef El::DistMatrix<double, El::STAR, El::VC> DistMatrix_STAR_VC;
typedef skylark::base::sparse_matrix_t<double> SparseMatrix;
#ifdef SKYLARK_HAVE_COMBBLAS
typedef SpDCCols< size_t, double > col_t;
typedef SpParMat< size_t, double, col_t > DistSparseMatrix;
#endif


enum matrix_type_t {
    MATRIX_TYPE_ERROR,
    MATRIX,                     /**< Dense Elemental matrix */
    SHARED_MATRIX,              /**< Same matrix on all processors: STAR-STAR */
    ROOT_MATRIX,                /**< One rank holds the matrix: CIRC-CIRC */
    DIST_MATRIX,                /**< Distributed Elemental matrix (MC-MR) */
    DIST_MATRIX_VC_STAR,
    DIST_MATRIX_VR_STAR,
    DIST_MATRIX_STAR_VC,
    DIST_MATRIX_STAR_VR,
    DIST_SPARSE_MATRIX,          /**< Sparse matrix (CombBLAS) */
    SPARSE_MATRIX                /**< Sparse local matrix */
};

matrix_type_t str2matrix_type(const char *str);
boost::any skylark_void2any(const char *type, void *obj);

#endif // SKYLARK_MATRIX_TYPES_HPP
