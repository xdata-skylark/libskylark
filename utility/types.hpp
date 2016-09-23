#ifndef SKYLARK_UTILITY_TYPES_HPP
#define SKYLARK_UTILITY_TYPES_HPP

namespace skylark {


/** Short types name for use in macros */
namespace mdtypes {

typedef El::Matrix<double> matrix_t;
typedef El::ElementalMatrix<double> el_matrix_t;
typedef base::sparse_matrix_t<double> sparse_matrix_t;
typedef El::DistMatrix<double> dist_matrix_t;
typedef El::DistMatrix<double, El::STAR, El::STAR> shared_matrix_t;
typedef El::DistMatrix<double, El::CIRC, El::CIRC> root_matrix_t;
typedef El::DistMatrix<double, El::VC, El::STAR> dist_matrix_vc_star_t;
typedef El::DistMatrix<double, El::VR, El::STAR> dist_matrix_vr_star_t;
typedef El::DistMatrix<double, El::STAR, El::VC> dist_matrix_star_vc_t;
typedef El::DistMatrix<double, El::STAR, El::VR> dist_matrix_star_vr_t;

// TODO
//#ifdef SKYLARK_HAVE_COMBBLAS
//typedef SpParMat<size_t, double, SpDCCols<size_t, double> >
//cb_dist_sparse_matrix_t;
//#endif

}

namespace mftypes {

typedef El::Matrix<float> matrix_t;
typedef El::ElementalMatrix<float> el_matrix_t;
typedef base::sparse_matrix_t<float> sparse_matrix_t;
typedef El::DistMatrix<float> dist_matrix_t;
typedef El::DistMatrix<float, El::STAR, El::STAR> shared_matrix_t;
typedef El::DistMatrix<float, El::CIRC, El::CIRC> root_matrix_t;
typedef El::DistMatrix<float, El::VC, El::STAR> dist_matrix_vc_star_t;
typedef El::DistMatrix<float, El::VR, El::STAR> dist_matrix_vr_star_t;
typedef El::DistMatrix<float, El::STAR, El::VC> dist_matrix_star_vc_t;
typedef El::DistMatrix<float, El::STAR, El::VR> dist_matrix_star_vr_t;

// TODO
//#ifdef SKYLARK_HAVE_COMBBLAS
//typedef SpParMat<size_t, float, SpDCCols<size_t, float> >
//cb_dist_sparse_matrix_t;
//#endif

}

}

#endif // SKYLARK_UTILITY_TYPES_HPP
