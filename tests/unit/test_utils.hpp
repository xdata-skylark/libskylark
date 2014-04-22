#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "../../base/svd.hpp"

#if SKYLARK_HAVE_BOOST
#include <boost/test/minimal.hpp>
#endif

#if SKYLARK_HAVE_ELEMENTAL
#include <elemental.hpp>
#endif


#if SKYLARK_HAVE_ELEMENTAL

template<typename MatrixType>
MatrixType operator-(MatrixType& A, MatrixType& B) {
    MatrixType C;
    elem::Copy(A, C);
    elem::Axpy(-1.0, B, C);
    return C;
}


template<typename MatrixType>
bool equal(MatrixType& A, MatrixType& B,  double threshold=1.e-4) {
    MatrixType C = A - B;
    double diff_norm = elem::Norm(C);
    if (diff_norm < threshold) {
        return true;
    }
    return false;
}


template<typename InputMatrixType,
         typename LeftSingularVectorsMatrixType,
         typename SingularValuesMatrixType,
         typename RightSingularVectorsMatrixType>
bool equal_svd_product(InputMatrixType& A,
    LeftSingularVectorsMatrixType& U,
    SingularValuesMatrixType& S,
    RightSingularVectorsMatrixType& V,
    double threshold=1e-4) {

    elem::DistMatrix<double> S_CIRC_CIRC = S;
    std::vector<double> values(S_CIRC_CIRC.Buffer(),
        S_CIRC_CIRC.Buffer() + S_CIRC_CIRC.Height());
    elem::Diagonal(S_CIRC_CIRC, values);
    elem::DistMatrix<double> S_MC_MR = S_CIRC_CIRC;

    elem::DistMatrix<double> A_MC_MR = A;
    elem::DistMatrix<double> U_MC_MR = U;
    elem::DistMatrix<double> V_MC_MR = V;
    elem::DistMatrix<double> US_MC_MR;
    elem::DistMatrix<double> USVt_MC_MR;

    US_MC_MR.Resize(U.Height(), S_CIRC_CIRC.Width());
    elem::Zero(US_MC_MR);
    USVt_MC_MR.Resize(U.Height(), V.Height());

    elem::Zero(USVt_MC_MR);
    elem::Gemm(elem::NORMAL, elem::NORMAL,    1.0, U_MC_MR,
        S_MC_MR, 0.0, US_MC_MR);
    elem::Gemm(elem::NORMAL, elem::TRANSPOSE, 1.0, US_MC_MR,
        V_MC_MR, 0.0, USVt_MC_MR);

    return equal(A_MC_MR, USVt_MC_MR, threshold);
}


#if SKYLARK_HAVE_BOOST

void check(elem::DistMatrix<double>& A,
    double threshold=1e-4) {
    elem::DistMatrix<double> U, V;
    elem::DistMatrix<double, elem::VR, elem::STAR> S_VR_STAR;
    skylark::base::svd(A, U, S_VR_STAR, V);
    bool passed = equal_svd_product(A, U, S_VR_STAR, V, threshold);
    if (!passed) {
        BOOST_FAIL("Failure in [MC, MR] case");
    }

}


template<elem::Distribution ColDist>
void check(elem::DistMatrix<double, ColDist, elem::STAR>& A,
    double threshold=1e-4) {
    elem::DistMatrix<double, ColDist, elem::STAR> A_CD_STAR, U_CD_STAR;
    elem::DistMatrix<double, elem::STAR, elem::STAR> S_STAR_STAR, V_STAR_STAR;
    A_CD_STAR = A;
    skylark::base::svd(A_CD_STAR, U_CD_STAR, S_STAR_STAR, V_STAR_STAR);
    bool passed = equal_svd_product(A_CD_STAR,
        U_CD_STAR, S_STAR_STAR, V_STAR_STAR, threshold);
    if (!passed) {
        BOOST_FAIL("Failure in [VC/VR, *] case");
    }
}

template<elem::Distribution RowDist>
void check(elem::DistMatrix<double, elem::STAR, RowDist>& A,
    double threshold=1e-4) {
    elem::DistMatrix<double, RowDist, elem::STAR> V_RD_STAR;
    elem::DistMatrix<double, elem::STAR, elem::STAR> S_STAR_STAR, U_STAR_STAR;
    skylark::base::svd(A, U_STAR_STAR, S_STAR_STAR, V_RD_STAR);
    bool passed = equal_svd_product(A,
        U_STAR_STAR, S_STAR_STAR, V_RD_STAR, threshold);
    if (!passed) {
        BOOST_FAIL("Failure in [*, VC/VR] case");
    }

}


#endif // SKYLARK_HAVE_BOOST

#endif // SKYLARK_HAVE_ELEMENTAL

#endif // TEST_UTILS_HPP
